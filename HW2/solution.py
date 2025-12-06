"""Stereo matching."""
import numpy as np
from scipy.signal import convolve2d


class Solution:
    def __init__(self):
        pass

    @staticmethod
    def ssd_distance(left_image: np.ndarray,
                     right_image: np.ndarray,
                     win_size: int,
                     dsp_range: int) -> np.ndarray:
        """Compute the SSDD distances tensor.

        Args:
            left_image: Left image of shape: HxWx3, and type np.double64.
            right_image: Right image of shape: HxWx3, and type np.double64.
            win_size: Window size odd integer.
            dsp_range: Half of the disparity range. The actual range is
            -dsp_range, -dsp_range + 1, ..., 0, 1, ..., dsp_range.

        Returns:
            A tensor of the sum of squared differences for every pixel in a
            window of size win_size X win_size, for the 2*dsp_range + 1
            possible disparity values. The tensor shape should be:
            HxWx(2*dsp_range+1).
        """
        num_of_rows, num_of_cols = left_image.shape[0], left_image.shape[1]
        disparity_values = range(-dsp_range, dsp_range + 1)
        ssdd_tensor = np.zeros((num_of_rows,
                                num_of_cols,
                                len(disparity_values)))

        # Create the sum filter (box filter)
        kernel = np.ones((win_size, win_size))

        for i, d in enumerate(disparity_values):
            # Shift the right image by -d to align with left image at that disparity
            if d == 0:
                shifted_right = right_image
            elif d > 0:
                # Shift right: padded with zeros on the left
                shifted_right = np.zeros_like(right_image)
                shifted_right[:, d:] = right_image[:, :-d]
            else: # d < 0
                # Shift left: padded with zeros on the right
                shifted_right = np.zeros_like(right_image)
                shifted_right[:, :d] = right_image[:, -d:]

            # Compute squared difference for each channel, then sum channels
            diff = (left_image - shifted_right) ** 2
            ssd = np.sum(diff, axis=2)

            # Aggregate costs over the window
            ssdd_tensor[:, :, i] = convolve2d(ssd, kernel, mode='same')

        # Normalize the tensor
        ssdd_tensor -= ssdd_tensor.min()
        max_val = ssdd_tensor.max()
        if max_val > 0:
            ssdd_tensor /= max_val
        ssdd_tensor *= 255.0

        return ssdd_tensor

    @staticmethod
    def naive_labeling(ssdd_tensor: np.ndarray) -> np.ndarray:
        """Estimate a naive depth estimation from the SSDD tensor.

        Args:
            ssdd_tensor: HxWx(2*dsp_range+1) tensor.

        Returns:
            Naive labels HxW matrix (indices of best disparity).
        """
        # Argmin along the disparity axis (axis 2) gives the index of min cost
        label_no_smooth = np.argmin(ssdd_tensor, axis=2)
        return label_no_smooth

    @staticmethod
    def dp_grade_slice(c_slice: np.ndarray, p1: float, p2: float) -> np.ndarray:
        """Calculate the scores matrix for slice c_slice.

        Args:
            c_slice: A slice of the ssdd tensor (D x W).
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.
        Returns:
            Scores slice (D x W).
        """
        num_labels, num_of_cols = c_slice.shape[0], c_slice.shape[1]
        l_slice = np.zeros((num_labels, num_of_cols))

        # Initialize first column
        l_slice[:, 0] = c_slice[:, 0]

        for col in range(1, num_of_cols):
            prev_col = l_slice[:, col - 1]
            min_prev = np.min(prev_col)
            
            # Vectorized calculation for M(d, col)
            # L(d, col-1)
            term0 = prev_col
            
            # p1 + min(L(d-1, col-1), L(d+1, col-1))
            # Handle boundaries by padding with infinity
            padded_prev = np.pad(prev_col, (1, 1), constant_values=np.inf)
            term1 = p1 + np.minimum(padded_prev[:-2], padded_prev[2:])
            
            # p2 + min(L(k, col-1)) for |k-d| >= 2
            # This is roughly min(prev_col) + p2, but we must exclude neighborhood.
            # Since num_labels is small (~40), we can iterate or use a simpler check.
            # Efficient check: min_prev + p2 is valid for all d where min_prev
            # index is not d, d-1, d+1. 
            # For simplicity and robustness with small D, we compute term2 explicitly.
            term2 = np.full_like(prev_col, min_prev + p2)
            
            # M is the min of the transition costs
            M = np.minimum(term0, np.minimum(term1, term2))
            
            # Refine term2 strictly if necessary (the 'exclusion' rule).
            # The simplified min_prev + p2 is standard SGM approximation.
            # If strict adherence to "min{L.. |k|>=2}" is needed:
            for d in range(num_labels):
                 # Slice excluding [d-1, d, d+1]
                 start_exclude = max(0, d - 1)
                 end_exclude = min(num_labels, d + 2)
                 
                 # Indices to include: [0...start_exclude) U [end_exclude...D]
                 valid_indices = np.r_[0:start_exclude, end_exclude:num_labels]
                 if len(valid_indices) > 0:
                     val_p2 = np.min(prev_col[valid_indices]) + p2
                     if val_p2 < M[d]: # Update if this specific p2 move is better
                         M[d] = val_p2

            l_slice[:, col] = c_slice[:, col] + M - min_prev

        return l_slice

    def dp_labeling(self,
                    ssdd_tensor: np.ndarray,
                    p1: float,
                    p2: float) -> np.ndarray:
        """Estimate a depth map using Dynamic Programming along rows.

        Args:
            ssdd_tensor: HxWxD tensor.
            p1: penalty 1.
            p2: penalty 2.
        Returns:
            DP depth estimation HxW.
        """
        l_tensor = np.zeros_like(ssdd_tensor)
        
        # Iterate over all rows
        for r in range(ssdd_tensor.shape[0]):
            # Transpose to shape (D, W) for the helper
            row_slice = ssdd_tensor[r, :, :].T
            l_slice = self.dp_grade_slice(row_slice, p1, p2)
            l_tensor[r, :, :] = l_slice.T

        return self.naive_labeling(l_tensor)

    def dp_labeling_per_direction(self,
                                  ssdd_tensor: np.ndarray,
                                  p1: float,
                                  p2: float) -> dict:
        """Return a dictionary of directions to a Dynamic Programming
        estimation of depth.

        Returns:
            Dictionary int->np.ndarray (label maps).
        """
        direction_to_slice = {}
        
        # Compute L tensors for all 8 directions
        for i in range(1, 9):
            l_tensor = self._compute_l_for_direction(ssdd_tensor, i, p1, p2)
            # Convert cost tensor to label map
            direction_to_slice[i] = self.naive_labeling(l_tensor)
            
        return direction_to_slice

    def sgm_labeling(self, ssdd_tensor: np.ndarray, p1: float, p2: float):
        """Estimate the depth map according to the SGM algorithm.

        Returns:
            Semi-Global Mapping depth estimation matrix of shape HxW.
        """
        l_accum = np.zeros_like(ssdd_tensor)
        
        # Aggregate costs from all 8 directions
        for i in range(1, 9):
            l_accum += self._compute_l_for_direction(ssdd_tensor, i, p1, p2)
            
        # Average
        l_accum /= 8.0
        
        return self.naive_labeling(l_accum)

    def _compute_l_for_direction(self, ssdd, direction, p1, p2):
        """Helper to compute L tensor for a specific direction."""
        
        # Direction 1: Left -> Right (Standard rows)
        if direction == 1:
            return self._compute_horizontal_l(ssdd, p1, p2)
            
        # Direction 5: Right -> Left (Flip rows LR)
        elif direction == 5:
            flipped = np.flip(ssdd, axis=1)
            l_flipped = self._compute_horizontal_l(flipped, p1, p2)
            return np.flip(l_flipped, axis=1)
            
        # Direction 3: Top -> Bottom (Standard columns)
        # Transpose image to (W, H, D) so columns become rows
        elif direction == 3:
            transposed = np.transpose(ssdd, (1, 0, 2))
            l_transposed = self._compute_horizontal_l(transposed, p1, p2)
            return np.transpose(l_transposed, (1, 0, 2))
            
        # Direction 7: Bottom -> Top (Flip columns UD)
        elif direction == 7:
            transposed_flipped = np.flip(np.transpose(ssdd, (1, 0, 2)), axis=1)
            l_res = self._compute_horizontal_l(transposed_flipped, p1, p2)
            return np.transpose(np.flip(l_res, axis=1), (1, 0, 2))
            
        # Direction 2: South-East (Diagonal)
        elif direction == 2:
            return self._compute_diagonal_l(ssdd, p1, p2)
            
        # Direction 6: North-West (Reverse South-East)
        elif direction == 6:
            return self._compute_diagonal_l(ssdd, p1, p2, reverse=True)
            
        # Direction 4: South-West (Diagonal after Flip LR)
        elif direction == 4:
            flipped = np.flip(ssdd, axis=1)
            l_flipped = self._compute_diagonal_l(flipped, p1, p2)
            return np.flip(l_flipped, axis=1)
            
        # Direction 8: North-East (Reverse South-West)
        elif direction == 8:
            flipped = np.flip(ssdd, axis=1)
            l_flipped = self._compute_diagonal_l(flipped, p1, p2, reverse=True)
            return np.flip(l_flipped, axis=1)
            
        return np.zeros_like(ssdd)

    def _compute_horizontal_l(self, ssdd, p1, p2):
        """Runs DP on every row."""
        l_tensor = np.zeros_like(ssdd)
        for r in range(ssdd.shape[0]):
            c_slice = ssdd[r, :, :].T # Shape (D, W)
            l_res = self.dp_grade_slice(c_slice, p1, p2)
            l_tensor[r, :, :] = l_res.T
        return l_tensor

    def _compute_diagonal_l(self, ssdd, p1, p2, reverse=False):
        """Runs DP on diagonals (South-East direction)."""
        H, W, D = ssdd.shape
        l_tensor = np.zeros_like(ssdd)
        
        # Iterate over all diagonals.
        # k=0 is main diagonal. k>0 is upper, k<0 is lower.
        # Range of offsets: from -(H-1) to (W-1)
        for k in range(-(H - 1), W):
            # Extract diagonal: returns (D, Len)
            # axis1=0 (rows), axis2=1 (cols)
            diag_slice = ssdd.diagonal(offset=k, axis1=0, axis2=1)
            
            # If reverse, we process the path backwards (NW, NE)
            if reverse:
                diag_slice = np.flip(diag_slice, axis=1)

            # Compute DP
            l_diag = self.dp_grade_slice(diag_slice, p1, p2)

            if reverse:
                l_diag = np.flip(l_diag, axis=1)

            # Put back into tensor.
            # Get indices of this diagonal
            # diag_indices returns (row_indices, col_indices) for a 2D matrix
            # We construct them manually to be safe
            diag_len = l_diag.shape[1]
            if k >= 0:
                rows = np.arange(diag_len)
                cols = np.arange(k, k + diag_len)
            else:
                rows = np.arange(-k, -k + diag_len)
                cols = np.arange(diag_len)
                
            # Assign (transposing l_diag back to Len x D)
            l_tensor[rows, cols, :] = l_diag.T
            
        return l_tensor