# HW2 - Stereo Matching

This project implements stereo matching algorithms for depth estimation from stereo image pairs, including:
- Sum of Squared Differences (SSD) distance calculation
- Naive depth labeling
- Dynamic Programming (DP) based depth estimation
- Semi-Global Matching (SGM) algorithm

## Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

## Setup Instructions

### Step 1: Navigate to the HW2 Directory

Open a terminal/command prompt and navigate to the HW2 folder:

```bash
cd HW2
```

### Step 2: Create a Virtual Environment

Choose the appropriate command for your operating system:

#### Windows (PowerShell)
```powershell
python -m venv venv
```

#### Windows (Command Prompt)
```cmd
python -m venv venv
```

#### macOS/Linux
```bash
python3 -m venv venv
```

If `python` or `python3` doesn't work, try `py` on Windows:
```powershell
py -m venv venv
```

### Step 3: Activate the Virtual Environment

#### Windows (PowerShell)
```powershell
.\venv\Scripts\Activate.ps1
```

#### Windows (Command Prompt)
```cmd
venv\Scripts\activate
```

#### macOS/Linux
```bash
source venv/bin/activate
```

After activation, you should see `(venv)` at the beginning of your command prompt.

### Step 4: Install Dependencies

With the virtual environment activated, install the required packages:

```bash
pip install -r requirements.txt
```

This will install:
- numpy
- scipy
- matplotlib
- opencv-python
- pillow

### Step 5: Verify Assets

Make sure the `assets` folder contains the required images:
- `image_left.png`
- `image_right.png`
- `my_left.jpg` (optional, for custom images)
- `my_right.jpg` (optional, for custom images)

## Running the Code

With the virtual environment activated, run:

```bash
python main.py
```

Or on some systems:

```bash
python3 main.py
```

The script will:
1. Load stereo image pairs
2. Compute SSD distances
3. Generate depth maps using various algorithms (naive, DP, SGM)
4. Display visualization plots using matplotlib

## Project Structure

```
HW2/
├── main.py              # Main execution script
├── solution.py          # Implementation of stereo matching algorithms
├── requirements.txt     # Python dependencies
├── assets/              # Image files for stereo matching
│   ├── image_left.png
│   ├── image_right.png
│   ├── my_left.jpg
│   └── my_right.jpg
└── README.md           # This file
```

## Deactivating the Virtual Environment

When you're done working, you can deactivate the virtual environment:

```bash
deactivate
```

## Notes

- The code uses matplotlib to display results, so make sure you have a display available or configure matplotlib for headless operation if needed
- Processing time depends on image size and disparity range settings. It's best to use images under 500x500 pixels for runtimes smaller than 10 minutes per image pairs.
- You can modify parameters (COST1, COST2, WIN_SIZE, DISPARITY_RANGE) in `main.py` to experiment with different settings