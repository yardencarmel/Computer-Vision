"""Projective Homography and Panorama Solution."""
import numpy as np

from typing import Tuple
from random import sample
from collections import namedtuple


from numpy.linalg import svd
from scipy.interpolate import griddata

PadStruct = namedtuple('PadStruct',
                       ['pad_up', 'pad_down', 'pad_right', 'pad_left'])


class Solution:
    """Implement Projective Homography and Panorama Solution."""
    def __init__(self):
        pass

    @staticmethod
    def compute_homography_naive(match_p_src: np.ndarray,
                                 match_p_dst: np.ndarray) -> np.ndarray:
        """Compute a Homography in the Naive approach, using SVD decomposition.

        Args:
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.

        Returns:
            Homography from source to destination, 3x3 numpy array.
        """
        # Extract number of point correspondences
        num_points = match_p_src.shape[1]
        
        # Build the constraint matrix A for DLT algorithm
        # Each point correspondence gives 2 equations, so A is 2N x 9
        A = np.zeros((2 * num_points, 9))
        
        for i in range(num_points):
            x_src, y_src = match_p_src[0, i], match_p_src[1, i]
            x_dst, y_dst = match_p_dst[0, i], match_p_dst[1, i]
            
            # First equation (for x coordinate)
            A[2 * i, :] = [0, 0, 0, -x_src, -y_src, -1, y_dst * x_src, y_dst * y_src, y_dst]
            # Second equation (for y coordinate)
            A[2 * i + 1, :] = [x_src, y_src, 1, 0, 0, 0, -x_dst * x_src, -x_dst * y_src, -x_dst]
        
        # Solve Ah = 0 using SVD
        # The solution is the right singular vector corresponding to the smallest singular value
        _, _, Vt = svd(A)
        h = Vt[-1, :]  # Last row of Vt (corresponds to smallest singular value)
        
        # Reshape h into 3x3 homography matrix
        homography = h.reshape(3, 3)
        
        # Normalize the homography (divide by h[2,2])
        if homography[2, 2] != 0:
            homography = homography / homography[2, 2]

        return homography

    @staticmethod
    def compute_forward_homography_slow(
            homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute a Forward-Homography in the Naive approach, using loops.

        Iterate over the rows and columns of the source image, and compute
        the corresponding point in the destination image using the
        projective homography. Place each pixel value from the source image
        to its corresponding location in the destination image.
        Don't forget to round the pixel locations computed using the
        homography.

        Args:
            homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination
            image height, width and color dimensions.

        Returns:
            The forward homography of the source image to its destination.
        """
        # Initialize destination image
        new_image = np.zeros(dst_image_shape, dtype=src_image.dtype)
        
        # Get source image dimensions
        src_height, src_width = src_image.shape[0], src_image.shape[1]
        
        # Iterate over all pixels in source image
        for y_src in range(src_height):
            for x_src in range(src_width):
                # Create homogeneous coordinates for source pixel
                # Note: x is column, y is row (1-indexed in image coordinates, but we use 0-indexed)
                src_point = np.array([x_src + 1, y_src + 1, 1])
                
                # Transform using homography
                dst_point_homogeneous = homography @ src_point
                
                # Normalize homogeneous coordinates
                if dst_point_homogeneous[2] != 0:
                    dst_point_homogeneous = dst_point_homogeneous / dst_point_homogeneous[2]
                
                # Extract x and y coordinates (convert to 0-indexed)
                x_dst = int(round(dst_point_homogeneous[0])) - 1
                y_dst = int(round(dst_point_homogeneous[1])) - 1
                
                # Check if destination coordinates are within bounds
                if 0 <= x_dst < dst_image_shape[1] and 0 <= y_dst < dst_image_shape[0]:
                    # Place pixel value from source to destination
                    new_image[y_dst, x_dst] = src_image[y_src, x_src]
        
        return new_image

    @staticmethod
    def compute_forward_homography_fast(
            homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute a Forward-Homography in a fast approach, WITHOUT loops.

        (1) Create a meshgrid of columns and rows.
        (2) Generate a matrix of size 3x(H*W) which stores the pixel locations
        in homogeneous coordinates.
        (3) Transform the source homogeneous coordinates to the target
        homogeneous coordinates with a simple matrix multiplication and
        apply the normalization you've seen in class.
        (4) Convert the coordinates into integer values and clip them
        according to the destination image size.
        (5) Plant the pixels from the source image to the target image according
        to the coordinates you found.

        Args:
            homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination.
            image height, width and color dimensions.

        Returns:
            The forward homography of the source image to its destination.
        """

        # Get source image dimensions
        src_rows, src_cols = src_image.shape[:2]
        dst_rows, dst_cols = dst_image_shape[:2]

        # (1) Create a meshgrid of columns and rows
        src_cols_grid, src_rows_grid = np.meshgrid(
            np.arange(1, src_cols + 1),  # 1-indexed x coordinates
            np.arange(1, src_rows + 1)    # 1-indexed y coordinates
        )

        # (2) Generate a matrix of size 3x(H*W) which stores the pixel locations
        # in homogeneous coordinates
        num_pixels = src_rows * src_cols
        homogeneous_coords = np.zeros((3, num_pixels))
        homogeneous_coords[0, :] = src_cols_grid.flatten()  # x coordinates
        homogeneous_coords[1, :] = src_rows_grid.flatten()  # y coordinates
        homogeneous_coords[2, :] = 1  # homogeneous coordinate

        # (3) Transform the source homogeneous coordinates to the target
        # homogeneous coordinates with a simple matrix multiplication and
        # apply the normalization
        transformed_coords = homography @ homogeneous_coords

        # Normalize by the third coordinate (w)
        transformed_coords = transformed_coords / transformed_coords[2, :]

        # Extract x and y coordinates
        dst_x_coords = transformed_coords[0, :]
        dst_y_coords = transformed_coords[1, :]

        # (4) Convert the coordinates into integer values and clip them
        # according to the destination image size
        # Convert to 0-indexed and round
        dst_x_int = np.round(dst_x_coords).astype(int) - 1
        dst_y_int = np.round(dst_y_coords).astype(int) - 1

        # Clip to valid image bounds
        dst_x_int = np.clip(dst_x_int, 0, dst_cols - 1)
        dst_y_int = np.clip(dst_y_int, 0, dst_rows - 1)

        # (5) Plant the pixels from the source image to the target image
        # according to the coordinates you found
        new_image = np.zeros(dst_image_shape, dtype=src_image.dtype)

        # Reshape source image pixels
        src_pixels = src_image.reshape(num_pixels, -1)  # Shape: (H*W, channels)

        # Use advanced indexing to place pixels
        new_image[dst_y_int, dst_x_int] = src_pixels

        return new_image

    @staticmethod
    def test_homography(homography: np.ndarray,
                        match_p_src: np.ndarray,
                        match_p_dst: np.ndarray,
                        max_err: float) -> Tuple[float, float]:
        """Calculate the quality of the projective transformation model.

        Args:
            homography: 3x3 Projective Homography matrix.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.

        Returns:
            A tuple containing the following metrics to quantify the
            homography performance:
            fit_percent: The probability (between 0 and 1) validly mapped src
            points (inliers).
            dist_mse: Mean square error of the distances between validly
            mapped src points, to their corresponding dst points (only for
            inliers). In edge case where the number of inliers is zero,
            return dist_mse = 10 ** 9.
        """

        # Transform source points to destination using homography
        num_points = match_p_src.shape[1]
        
        # Convert source points to homogeneous coordinates (3xN)
        src_homogeneous = np.vstack([match_p_src, np.ones((1, num_points))])
        
        # Transform using homography
        dst_transformed_homogeneous = homography @ src_homogeneous
        
        # Normalize homogeneous coordinates
        dst_transformed_homogeneous = dst_transformed_homogeneous / dst_transformed_homogeneous[2, :]
        
        # Extract 2D coordinates (x, y)
        dst_transformed = dst_transformed_homogeneous[:2, :]
        
        # Calculate Euclidean distances between transformed points and actual destination points
        distances = np.sqrt(np.sum((dst_transformed - match_p_dst) ** 2, axis=0))
        
        # Find inliers (points within max_err)
        inliers_mask = distances <= max_err
        num_inliers = np.sum(inliers_mask)
        
        # Calculate fit_percent (probability of inliers)
        fit_percent = num_inliers / num_points if num_points > 0 else 0.0
        
        # Calculate dist_mse for inliers only
        if num_inliers > 0:
            dist_mse = np.mean(distances[inliers_mask] ** 2)
        else:
            dist_mse = 10 ** 9
        
        return fit_percent, dist_mse

    @staticmethod
    def meet_the_model_points(homography: np.ndarray,
                              match_p_src: np.ndarray,
                              match_p_dst: np.ndarray,
                              max_err: float) -> Tuple[np.ndarray, np.ndarray]:
        """Return which matching points meet the homography.

        Loop through the matching points, and return the matching points from
        both images that are inliers for the given homography.

        Args:
            homography: 3x3 Projective Homography matrix.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.
        Returns:
            A tuple containing two numpy nd-arrays, containing the matching
            points which meet the model (the homography). The first entry in
            the tuple is the matching points from the source image. That is a
            nd-array of size 2xD (D=the number of points which meet the model).
            The second entry is the matching points form the destination
            image (shape 2xD; D as above).
        """

        # Transform source points to destination using homography
        num_points = match_p_src.shape[1]
        
        # Convert source points to homogeneous coordinates (3xN)
        src_homogeneous = np.vstack([match_p_src, np.ones((1, num_points))])
        
        # Transform using homography
        dst_transformed_homogeneous = homography @ src_homogeneous
        
        # Normalize homogeneous coordinates
        dst_transformed_homogeneous = dst_transformed_homogeneous / dst_transformed_homogeneous[2, :]
        
        # Extract 2D coordinates (x, y)
        dst_transformed = dst_transformed_homogeneous[:2, :]
        
        # Calculate Euclidean distances between transformed points and actual destination points
        distances = np.sqrt(np.sum((dst_transformed - match_p_dst) ** 2, axis=0))
        
        # Find inliers (points within max_err)
        inliers_mask = distances <= max_err
        
        # Extract inlier points from both source and destination
        mp_src_meets_model = match_p_src[:, inliers_mask]
        mp_dst_meets_model = match_p_dst[:, inliers_mask]
        
        return mp_src_meets_model, mp_dst_meets_model

    def compute_homography(self,
                           match_p_src: np.ndarray,
                           match_p_dst: np.ndarray,
                           inliers_percent: float,
                           max_err: float) -> np.ndarray:
        """Compute homography coefficients using RANSAC to overcome outliers.

        Args:
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            inliers_percent: The expected probability (between 0 and 1) of
            correct match points from the entire list of match points.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.
        Returns:
            homography: Projective transformation matrix from src to dst.
        """

        # use class notations:
        w = inliers_percent
        # t = max_err
        # p = parameter determining the probability of the algorithm to succeed
        p = 0.99
        # the minimal probability of points which meets with the model
        d = 0.5
        # number of points sufficient to compute the model
        n = 4
        # number of RANSAC iterations (+1 to avoid the case where w=1)
        k = int(np.ceil(np.log(1 - p) / np.log(1 - w ** n))) + 1
        
        num_points = match_p_src.shape[1]
        best_homography = None
        best_inlier_count = 0
        
        # RANSAC iterations
        for iteration in range(k):
            # Randomly sample n points (minimum needed for homography)
            if num_points < n:
                # Not enough points, use all points
                sample_indices = np.arange(num_points)
            else:
                sample_indices = np.array(sample(range(num_points), n))
            
            # Extract sampled points
            sample_src = match_p_src[:, sample_indices]
            sample_dst = match_p_dst[:, sample_indices]
            
            # Compute homography from sampled points
            try:
                candidate_homography = self.compute_homography_naive(sample_src, sample_dst)
                
                # Test the homography on all points
                mp_src_inliers, mp_dst_inliers = self.meet_the_model_points(
                    candidate_homography, match_p_src, match_p_dst, max_err
                )
                
                # Count inliers
                num_inliers = mp_src_inliers.shape[1]
                
                # Update best homography if this one has more inliers
                if num_inliers > best_inlier_count:
                    best_inlier_count = num_inliers
                    best_homography = candidate_homography
                    
                    # If we have enough inliers (d * total points), recompute with all inliers
                    if num_inliers >= d * num_points:
                        # Recompute homography using all inliers for better accuracy
                        best_homography = self.compute_homography_naive(mp_src_inliers, mp_dst_inliers)
                    
            except:
                # If homography computation fails, continue to next iteration
                continue
        
        # If no good homography found, use the best one we have
        if best_homography is None:
            # Fallback: use all points
            best_homography = self.compute_homography_naive(match_p_src, match_p_dst)
        
        return best_homography

    @staticmethod
    def compute_backward_mapping(
            backward_projective_homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute backward mapping.

        (1) Create a mesh-grid of columns and rows of the destination image.
        (2) Create a set of homogenous coordinates for the destination image
        using the mesh-grid from (1).
        (3) Compute the corresponding coordinates in the source image using
        the backward projective homography.
        (4) Create the mesh-grid of source image coordinates.
        (5) For each color channel (RGB): Use scipy's interpolation.griddata
        with an appropriate configuration to compute the bi-cubic
        interpolation of the projected coordinates.

        Args:
            backward_projective_homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination shape.

        Returns:
            The source image backward warped to the destination coordinates.
        """

        # Get dimensions
        dst_rows, dst_cols = dst_image_shape[:2]
        src_rows, src_cols = src_image.shape[:2]
        
        # (1) Create a mesh-grid of columns and rows of the destination image
        dst_cols_grid, dst_rows_grid = np.meshgrid(
            np.arange(1, dst_cols + 1),  # 1-indexed x coordinates
            np.arange(1, dst_rows + 1)    # 1-indexed y coordinates
        )
        
        # (2) Create a set of homogeneous coordinates for the destination image
        num_dst_pixels = dst_rows * dst_cols
        dst_homogeneous = np.zeros((3, num_dst_pixels))
        dst_homogeneous[0, :] = dst_cols_grid.flatten()  # x coordinates
        dst_homogeneous[1, :] = dst_rows_grid.flatten()  # y coordinates
        dst_homogeneous[2, :] = 1  # homogeneous coordinate
        
        # (3) Compute the corresponding coordinates in the source image using
        # the backward projective homography
        src_transformed_homogeneous = backward_projective_homography @ dst_homogeneous
        
        # Normalize homogeneous coordinates
        src_transformed_homogeneous = src_transformed_homogeneous / src_transformed_homogeneous[2, :]
        
        # Extract x and y coordinates in source image space
        src_x_coords = src_transformed_homogeneous[0, :]
        src_y_coords = src_transformed_homogeneous[1, :]
        
        # (4) Create the mesh-grid of source image coordinates
        src_cols_grid, src_rows_grid = np.meshgrid(
            np.arange(1, src_cols + 1),  # 1-indexed x coordinates
            np.arange(1, src_rows + 1)    # 1-indexed y coordinates
        )
        
        # Flatten source coordinates for griddata
        src_points = np.column_stack([
            src_cols_grid.flatten(),  # x coordinates
            src_rows_grid.flatten()   # y coordinates
        ])
        
        # Points to interpolate at (transformed coordinates)
        xi = np.column_stack([src_x_coords, src_y_coords])
        
        # (5) For each color channel (RGB): Use scipy's interpolation.griddata
        # with an appropriate configuration to compute the bi-cubic
        # interpolation of the projected coordinates
        backward_warp = np.zeros(dst_image_shape, dtype=src_image.dtype)
        
        # Get source image pixel values (flattened)
        src_values = src_image.reshape(src_rows * src_cols, -1)  # Shape: (H*W, channels)
        
        # Interpolate for each channel
        for channel in range(src_image.shape[2]):    
            # Use cubic interpolation (bi-cubic)
            interpolated = griddata(
                src_points,           # Known points (source image coordinates)
                src_values[:, channel],  # Known values (source image pixel values)
                xi,                   # Points to interpolate at (transformed coordinates)
                method='cubic',       # Bi-cubic interpolation
                fill_value=0          # Fill value for points outside the source image
            )
            backward_warp[:, :, channel] = interpolated.reshape(dst_rows, dst_cols)
            
        return backward_warp

    @staticmethod
    def find_panorama_shape(src_image: np.ndarray,
                            dst_image: np.ndarray,
                            homography: np.ndarray
                            ) -> Tuple[int, int, PadStruct]:
        """Compute the panorama shape and the padding in each axes.

        Args:
            src_image: Source image expected to undergo projective
            transformation.
            dst_image: Destination image to which the source image is being
            mapped to.
            homography: 3x3 Projective Homography matrix.

        For each image we define a struct containing it's corners.
        For the source image we compute the projective transformation of the
        coordinates. If some of the transformed image corners yield negative
        indices - the resulting panorama should be padded with at least
        this absolute amount of pixels.
        The panorama's shape should be:
        dst shape + |the largest negative index in the transformed src index|.

        Returns:
            The panorama shape and a struct holding the padding in each axes (
            row, col).
            panorama_rows_num: The number of rows in the panorama of src to dst.
            panorama_cols_num: The number of columns in the panorama of src to
            dst.
            padStruct = a struct with the padding measures along each axes
            (row,col).
        """
        src_rows_num, src_cols_num, _ = src_image.shape
        dst_rows_num, dst_cols_num, _ = dst_image.shape
        src_edges = {}
        src_edges['upper left corner'] = np.array([1, 1, 1])
        src_edges['upper right corner'] = np.array([src_cols_num, 1, 1])
        src_edges['lower left corner'] = np.array([1, src_rows_num, 1])
        src_edges['lower right corner'] = \
            np.array([src_cols_num, src_rows_num, 1])
        transformed_edges = {}
        for corner_name, corner_location in src_edges.items():
            transformed_edges[corner_name] = homography @ corner_location
            transformed_edges[corner_name] /= transformed_edges[corner_name][-1]
        pad_up = pad_down = pad_right = pad_left = 0
        for corner_name, corner_location in transformed_edges.items():
            if corner_location[1] < 1:
                # pad up
                pad_up = max([pad_up, abs(corner_location[1])])
            if corner_location[0] > dst_cols_num:
                # pad right
                pad_right = max([pad_right,
                                 corner_location[0] - dst_cols_num])
            if corner_location[0] < 1:
                # pad left
                pad_left = max([pad_left, abs(corner_location[0])])
            if corner_location[1] > dst_rows_num:
                # pad down
                pad_down = max([pad_down,
                                corner_location[1] - dst_rows_num])
        panorama_cols_num = int(dst_cols_num + pad_right + pad_left)
        panorama_rows_num = int(dst_rows_num + pad_up + pad_down)
        pad_struct = PadStruct(pad_up=int(pad_up),
                               pad_down=int(pad_down),
                               pad_left=int(pad_left),
                               pad_right=int(pad_right))
        return panorama_rows_num, panorama_cols_num, pad_struct

    @staticmethod
    def add_translation_to_backward_homography(backward_homography: np.ndarray,
                                               pad_left: int,
                                               pad_up: int) -> np.ndarray:
        """Create a new homography which takes translation into account.

        Args:
            backward_homography: 3x3 Projective Homography matrix.
            pad_left: number of pixels that pad the destination image with
            zeros from left.
            pad_up: number of pixels that pad the destination image with
            zeros from the top.

        (1) Build the translation matrix from the pads.
        (2) Compose the backward homography and the translation matrix together.
        (3) Scale the homography as learnt in class.

        Returns:
            A new homography which includes the backward homography and the
            translation.
        """

        # (1) Build the translation matrix from the pads
        # We need to convert panorama coordinates to destination image coordinates
        # Panorama coordinates have padding, so we subtract the padding to get
        # destination image coordinates. Translation by (-pad_left, -pad_up):
        # [1  0  -pad_left]
        # [0  1  -pad_up  ]
        # [0  0  1        ]
        translation_matrix = np.array([
            [1, 0, -pad_left],
            [0, 1, -pad_up],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # (2) Compose the backward homography and the translation matrix together
        # We apply translation first (to convert panorama coords to dst coords),
        # then backward homography (to convert dst coords to src coords):
        # final_homography = backward_homography @ translation_matrix
        final_homography = backward_homography @ translation_matrix
        
        # (3) Scale the homography as learnt in class (normalize by h[2,2])
        if final_homography[2, 2] != 0:
            final_homography = final_homography / final_homography[2, 2]
        
        return final_homography

    def panorama(self,
                 src_image: np.ndarray,
                 dst_image: np.ndarray,
                 match_p_src: np.ndarray,
                 match_p_dst: np.ndarray,
                 inliers_percent: float,
                 max_err: float) -> np.ndarray:
        """Produces a panorama image from two images, and two lists of
        matching points, that deal with outliers using RANSAC.

        (1) Compute the forward homography and the panorama shape.
        (2) Compute the backward homography.
        (3) Add the appropriate translation to the homography so that the
        source image will plant in place.
        (4) Compute the backward warping with the appropriate translation.
        (5) Create the an empty panorama image and plant there the
        destination image.
        (6) place the backward warped image in the indices where the panorama
        image is zero.
        (7) Don't forget to clip the values of the image to [0, 255].


        Args:
            src_image: Source image expected to undergo projective
            transformation.
            dst_image: Destination image to which the source image is being
            mapped to.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            inliers_percent: The expected probability (between 0 and 1) of
            correct match points from the entire list of match points.
            max_err: A scalar that represents the maximum distance (in pixels)
            between the mapped src point to its corresponding dst point,
            in order to be considered as valid inlier.

        Returns:
            A panorama image.

        """
        
        # (1) Compute the forward homography and the panorama shape
        forward_homography = self.compute_homography(
            match_p_src, match_p_dst, inliers_percent, max_err
        )
        
        panorama_rows_num, panorama_cols_num, pad_struct = self.find_panorama_shape(
            src_image, dst_image, forward_homography
        )
        
        # (2) Compute the backward homography (inverse of forward homography)
        backward_homography = np.linalg.inv(forward_homography)
        # Normalize the backward homography
        if backward_homography[2, 2] != 0:
            backward_homography = backward_homography / backward_homography[2, 2]
        
        # (3) Add the appropriate translation to the homography so that the
        # source image will plant in place
        backward_homography_with_translation = self.add_translation_to_backward_homography(
            backward_homography, pad_struct.pad_left, pad_struct.pad_up
        )
        
        # (4) Compute the backward warping with the appropriate translation
        panorama_shape = (panorama_rows_num, panorama_cols_num, 3)
        backward_warped = self.compute_backward_mapping(
            backward_homography_with_translation, src_image, panorama_shape
        )
        
        # (5) Create an empty panorama image and plant there the destination image
        # Start with backward warped image (covers entire panorama)
        img_panorama = backward_warped.copy()
        
        # Place destination image at the correct position (with padding offset)
        # This will overwrite the backward warped image in the overlapping region
        dst_start_row = pad_struct.pad_up
        dst_end_row = dst_start_row + dst_image.shape[0]
        dst_start_col = pad_struct.pad_left
        dst_end_col = dst_start_col + dst_image.shape[1]
        
        img_panorama[dst_start_row:dst_end_row, dst_start_col:dst_end_col, :] = dst_image
        
        # (6) Place the backward warped image in the indices where the panorama
        # image is zero (i.e., where there's no destination image)
        # Actually, we already did this by starting with backward_warped and then
        # placing dst_image on top. But to be safe, let's also fill any remaining zeros
        # with backward warped values
        zero_mask = np.sum(img_panorama, axis=2) == 0
        num_zero_pixels = np.sum(zero_mask)
        if num_zero_pixels > 0:
            for channel in range(3):
                img_panorama[:, :, channel][zero_mask] = backward_warped[:, :, channel][zero_mask]
        
        # (7) Don't forget to clip the values of the image to [0, 255]
        return np.clip(img_panorama, 0, 255).astype(np.uint8)
