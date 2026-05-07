#!/usr/bin/env python3

# Early exit for help flags to avoid loading heavy imports
import sys
if '--help' in sys.argv or '-h' in sys.argv:
    print("""usage: tablet_session_visualizer.py [-h] [--f F]

Process session folder with multiple delta CSV files (one per frame).

optional arguments:
  -h, --help  show this help message and exit
  --f F       Path to the session folder: data2/session_1325501210916
              If not specified, folders in data2/ are shown for selection

Example:
          python3 tablet_session_visualizer.py --f data2/session_test
          
          """)
    sys.exit(0)

"""
Animate measured = ref + delta from maXTouch CSVs (multi-frame deltas, one CSV per frame) with Play/Pause control.

This version processes session folders where each frame is stored in a separate CSV file.
Data structure: data2/session_XXXXX/deltas_0.csv, deltas_TIMESTAMP.csv, etc.
All delta files are baseline-corrected by subtracting the first delta (deltas_0.csv).

Rules:
- Skip first row
- Ignore first two columns
- Remaining elements form one long vector
- First 52 elements = first column, next 52 the second column, etc. (column major)
- Rotate matrices 90° clockwise immediately after reshaping
- Click Play button or press Space to start/pause automatic animation (loops continuously)
- Press 'd' to advance frame, 'a' to go back a frame
- Click to select region with blob detection
- Multiple disconnected regions are tracked separately
"""

import argparse
import csv
from pathlib import Path
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.widgets import Button, CheckButtons
from scipy import ndimage
from matplotlib.patches import Polygon
from matplotlib.widgets import Slider
import time
from region_stats import compute_region_area, calculate_region_stats, calculate_summary_stats, print_region_stats, print_summary_stats

rows = 52
percentiles = "2,98"
ref = 'ref_normal.csv'
vmin=0
vmax=0
# Physical pixel spacing in mm (electrode pitch)
# Can be a single value for uniform spacing or (y_spacing, x_spacing) tuple
pixel_spacing_mm = 4.2  # 4.2mm electrode pitch for maXTouch sensor

def print_help():
    """Print help message with keyboard shortcuts."""
    help_text = """
╔═══════════════════════════════════════════════════════════════════════════╗
║                        MAXTOUCH CSV ANIMATOR - HELP                       ║
╠═══════════════════════════════════════════════════════════════════════════╣
║ ANIMATION:                                                                ║
║   Play Button    - Start/pause automatic frame animation (loops)         ║
║   Space          - Start/pause automatic frame animation (loops)         ║
║                                                                            ║
║ FRAME NAVIGATION:                                                         ║
║   d              - Advance to next frame                                  ║
║   a              - Go back to previous frame                              ║
║   Frame Timeline - Green markers indicate frames with positive values    ║
║                                                                            ║
║ FILE NAVIGATION:                                                          ║
║   Up Arrow       - Previous file in list                                 ║
║   Down Arrow     - Next file in list                                     ║
║                                                                            ║
║ REGION SELECTION:                                                         ║
║   Left Click     - Select region (flood fill from click point)           ║
║   Left Drag      - Select multiple regions by dragging                   ║
║   Right Click    - Deselect region (remove from selection)               ║
║   Right Drag     - Deselect multiple regions by dragging                 ║
║   u              - Undo last region selection                             ║
║   x              - Clear all selections                                   ║
║   p              - Print statistics for selected regions                  ║
║   s              - Save selected regions to disk                          ║
║                    (manual save only - auto-save disabled)               ║
║   Frame in filename - Checkbox to include frame number in saved filename ║
║                       (default: OFF, filename: session_regions.npz)      ║
║                                                                            ║
║ TOLERANCE ADJUSTMENT (for flood fill):                                   ║
║   + or =         - Increase tolerance (select larger regions)            ║
║   - or _         - Decrease tolerance (select smaller regions)           ║
║   Scroll Wheel   - Scroll up/down to increase/decrease tolerance         ║
║                                                                            ║
║ COLOR SCALE ADJUSTMENT:                                                   ║
║   z              - Decrease vmin (darken lower range)                    ║
║   c              - Increase vmin (brighten lower range)                  ║
║   ,              - Decrease vmax (darken upper range)                    ║
║   /              - Increase vmax (brighten upper range)                  ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""
    print(help_text)

def row_to_vector_after_stripping(row: List[str]) -> List[float]:
    # drop first two columns
    row = row[2:]
    out = []
    for tok in row:
        if tok == "" or tok is None:
            continue
        try:
            out.append(float(tok))
        except Exception:
            pass
    return out

def load_single_vector_from_csv(path: Path) -> np.ndarray:
    with path.open("r", newline="") as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        raise RuntimeError(f"{path} has no data rows after header")
    vec = row_to_vector_after_stripping(rows[1])
    if not vec:
        raise RuntimeError(f"{path} data row empty after stripping columns")
    return np.asarray(vec, dtype=np.float32)

def load_delta_vectors_from_csv(path: Path) -> List[np.ndarray]:
    frames = []
    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0:
                continue  # skip header
            vec = row_to_vector_after_stripping(row)
            if vec:
                frames.append(np.asarray(vec, dtype=np.float32))
    if not frames:
        raise RuntimeError(f"{path} contained no usable frames")
    return frames

def load_delta_vectors_from_folder(folder_path: Path) -> List[np.ndarray]:
    """
    Load delta vectors from a session folder where each frame is in a separate CSV file.
    Files are named like: deltas_0.csv, deltas_1325501213959.csv, etc.
    Returns list of delta vectors sorted by filename (timestamp).
    Optimized for performance with pandas if available, falls back to csv reader.
    """
    import glob
    import os

    # Find all delta CSV files in the folder
    delta_files = glob.glob(os.path.join(folder_path, "deltas_*.csv"))

    if not delta_files:
        raise RuntimeError(f"{folder_path} contained no delta CSV files")

    # Sort files by the numeric part of the filename
    def extract_timestamp(filepath):
        basename = os.path.basename(filepath)
        # Extract number from deltas_XXX.csv
        try:
            num_str = basename.replace("deltas_", "").replace(".csv", "")
            return int(num_str)
        except ValueError:
            return 0

    delta_files.sort(key=extract_timestamp)

    # Try to use pandas for faster CSV reading
    try:
        import pandas as pd
        frames = []
        for delta_file in delta_files:
            # Read CSV, skip first 2 columns, read only first data row
            df = pd.read_csv(delta_file, usecols=lambda x: x not in ['time', 'TIN'], nrows=1)
            vec = df.values.flatten()
            # Remove NaN values (from trailing empty columns)
            vec = vec[~np.isnan(vec)].astype(np.float32)
            if len(vec) > 0:
                frames.append(vec)
    except (ImportError, Exception):
        # Fallback to csv reader if pandas not available or fails
        frames = []
        for delta_file in delta_files:
            with open(delta_file, "r", newline="") as f:
                reader = csv.reader(f)
                for i, row in enumerate(reader):
                    if i == 0:
                        continue  # skip header
                    vec = row_to_vector_after_stripping(row)
                    if vec:
                        frames.append(np.asarray(vec, dtype=np.float32))
                    break  # Only read first data row from each file

    if not frames:
        raise RuntimeError(f"{folder_path} contained no usable delta frames")

    return frames

def vector_to_matrix_col_first_rot90(vec: np.ndarray, n_rows: int) -> np.ndarray:
    """Reshape vector (column major) then rotate 90° clockwise."""
    if vec.size % n_rows != 0:
        raise ValueError(f"vector length {vec.size} is not a multiple of n_rows {n_rows}")
    n_cols = vec.size // n_rows
    M = vec.reshape((n_rows, n_cols), order="F")
    M = np.rot90(M, k=-1)  # 90° clockwise
    return M

def clamp_extreme_values(data, lower_limit=-1000, upper_limit=1000):
    """
    Clamp extreme values to 0 to prevent abnormal values from affecting region detection.
    Values outside [lower_limit, upper_limit] are set to 0.
    
    Parameters:
    -----------
    data : np.ndarray
        Input data array
    lower_limit : float
        Lower threshold (default: -1000)
    upper_limit : float
        Upper threshold (default: +1000)
    
    Returns:
    --------
    np.ndarray
        Data with extreme values clamped to 0
    """
    clamped = data.copy()
    # Set values outside the range to 0
    clamped[(clamped < lower_limit) | (clamped > upper_limit)] = 0
    return clamped

def flood_fill_region(data, seed_y, seed_x, tolerance):
    """
    Perform flood fill (magic wand) from seed point.
    Returns binary mask of the selected region.
    Extreme values outside [-1000, +1000] are treated as 0.
    """
    # Clamp extreme values to prevent abnormal values from affecting selection
    data_clamped = clamp_extreme_values(data)
    
    h, w = data_clamped.shape
    if seed_y < 0 or seed_y >= h or seed_x < 0 or seed_x >= w:
        return None

    seed_value = data_clamped[seed_y, seed_x]
    mask = np.zeros((h, w), dtype=bool)
    to_check = [(seed_y, seed_x)]
    checked = set()

    while to_check:
        y, x = to_check.pop()
        if (y, x) in checked:
            continue
        if y < 0 or y >= h or x < 0 or x >= w:
            continue

        checked.add((y, x))

        if abs(data_clamped[y, x] - seed_value) <= tolerance:
            mask[y, x] = True
            # Add 8-connected neighbors (including diagonals)
            to_check.extend([
                (y+1, x), (y-1, x), (y, x+1), (y, x-1),  # 4-connected
                (y+1, x+1), (y+1, x-1), (y-1, x+1), (y-1, x-1)  # diagonals
            ])

    return mask

def separate_regions(mask, data=None, threshold=None, patch_size=6):
    """
    Separate a mask into individual connected components using 8-connectivity.
    
    Parameters:
    -----------
    mask : np.ndarray
        Binary mask to separate into regions
    data : np.ndarray, optional
        Data array to check for values (not used in this version)
    threshold : float, optional
        Threshold for rejecting regions (not used in this version)
    patch_size : int
        Size of the patch to check around each region (not used in this version)

    Returns:
    --------
    list of np.ndarray
        List of separate region masks
    """
    # Define 8-connected structure (includes diagonals)
    structure = np.array([[1, 1, 1],
                          [1, 1, 1],
                          [1, 1, 1]], dtype=bool)

    labeled, num_features = ndimage.label(mask, structure=structure)
    regions = []
    for i in range(1, num_features + 1):
        region_mask = (labeled == i)
        regions.append(region_mask)
    return regions


def zscore_threshold(data, z_threshold=2.0, mode='both'):
    """
    Create binary mask based on z-score threshold.
    Computes statistics on all data values.
    Extreme values outside [-1000, +1000] are treated as 0.

    Parameters:
    -----------
    data : np.ndarray
        2D array of sensor values
    z_threshold : float
        Number of standard deviations from mean
    mode : str
        'positive' - only above mean, 'negative' - only below mean, 'both' - either direction

    Returns:
    --------
    threshold : float
        The computed threshold value (positive for upper, negative for lower)
    """
    # Clamp extreme values to prevent abnormal values from affecting threshold calculation
    data_clamped = clamp_extreme_values(data)
    
    # Compute mean and std from clamped data
    mean = np.mean(data_clamped)
    std = np.std(data_clamped)

    if mode == 'positive':
        threshold = mean + z_threshold * std
    elif mode == 'negative':
        threshold = mean - z_threshold * std
    else:  # 'both' - use whichever direction has stronger signal
        # Check which direction has more extreme values (using clamped data)
        max_pos = np.max(data_clamped - mean)
        max_neg = np.abs(np.min(data_clamped - mean))

        if max_pos > max_neg:
            threshold = mean + z_threshold * std
        else:
            threshold = mean - z_threshold * std

    return threshold

def auto_select_blobs(data, z_threshold=2.0, min_size=3):
    """
    Automatically select prominent blobs in the data using Z-score thresholding.
    Selects blobs that deviate significantly from the mean.
    Extreme values outside [-1000, +1000] are treated as 0.

    Parameters:
    -----------
    data : np.ndarray
        2D sensor data
    z_threshold : float
        Z-score threshold
    min_size : int
        Minimum number of pixels for a blob to be kept

    Returns:
    --------
    mask : np.ndarray
        Binary mask of selected regions
    threshold : float
        The threshold value used
    """
    np.set_printoptions(threshold=np.inf)
    
    # Clamp extreme values to prevent abnormal values from affecting selection
    data_clamped = clamp_extreme_values(data)
    
    threshold = zscore_threshold(data_clamped, z_threshold, mode='both')
    # Select pixels that are far from mean in either direction
    mean = np.mean(data_clamped)
    if threshold > mean:
        # Positive blobs (above threshold)
        mask = data_clamped > threshold
    else:
        # Negative blobs (below threshold)
        mask = data_clamped < threshold

    # Remove small blobs using 8-connectivity
    structure = np.array([[1, 1, 1],
                          [1, 1, 1],
                          [1, 1, 1]], dtype=bool)
    labeled, num_features = ndimage.label(mask, structure=structure)
    for i in range(1, num_features + 1):
        region_mask = (labeled == i)

        # Check if too small
        if np.sum(region_mask) < min_size:
            mask[region_mask] = False
            continue

    return mask, threshold

def natural_sort_key(text):
    """
    Generate a key for natural (human) sorting.
    Converts numbers in strings to integers for proper numerical sorting.
    Example: ['file1.txt', 'file10.txt', 'file2.txt'] -> ['file1.txt', 'file2.txt', 'file10.txt']
    """
    import re
    def atoi(text):
        return int(text) if text.isdigit() else text.lower()

    return [atoi(c) for c in re.split(r'(\d+)', text)]

def load_data_file(file_path, _error_shown_cache={}, _data_cache={}):
    """Load data from a session folder with multiple delta CSV files."""
    session_folder = Path(file_path)

    # Look for ref.csv in the session folder first
    ref_path = session_folder / ref
    if not ref_path.exists():
        # Fall back to current directory
        ref_path = Path(ref)

    # Create cache key for this folder
    cache_key = str(session_folder.resolve())

    # Check if data is already cached
    if cache_key in _data_cache:
        return _data_cache[cache_key]

    # Check if the ref file exists
    if not ref_path.exists():
        print(f"Error: Reference file '{ref}' not found in '{session_folder}' or current directory.")
        print("Please make sure 'ref.csv' exists in the session folder or in the current directory.")
        return None

    # Check if the session folder exists
    if not session_folder.exists():
        print(f"Error: Session folder '{file_path}' not found.")
        print("Please check the folder path.")
        return None

    # Check if it's actually a directory
    if not session_folder.is_dir():
        print(f"Error: '{file_path}' is not a directory.")
        print("Please provide a path to a session folder (e.g., data2/session_XXXXX).")
        return None

    ref_vec = load_single_vector_from_csv(ref_path)
    delta_vecs = load_delta_vectors_from_folder(session_folder)

    ref_mat = vector_to_matrix_col_first_rot90(ref_vec, 52)
    H, W = ref_mat.shape

    delta_mats = []
    for v in delta_vecs:
        if v.size != ref_vec.size:
            break
        delta_mats.append(vector_to_matrix_col_first_rot90(v, 52))

    if not delta_mats:
        print(f"\nError: Incompatible data format in '{file_path}'")
        return None

    delta_mats = np.stack(delta_mats, axis=0)

    # Then add to reference to get measured values
    delta_mats = ref_mat[np.newaxis, :, :] + delta_mats

    # Subtract the first delta from all deltas (baseline correction)
    measured = delta_mats - delta_mats[0]
    
    # print ('MEASURED')
    # print (measured.shape)
    # print (measured[0])
    # print (measured[39])

    # Check if max value exceeds hard-coded normalization value
    actual_max = np.max(measured)
    max_value = 290  # hard coded value

    # if actual_max > max_value:
    #     # Check if we've already shown the error for this file
    #     if cache_key not in _error_shown_cache:
    #         _error_shown_cache[cache_key] = True

    #         # Print to console
    #         print("\n" + "="*60)
    #         print("ERROR: Data exceeds expected maximum value")
    #         print("="*60)
    #         print(f"\nFile: {file_path}")
    #         print(f"Expected maximum: {max_value}")
    #         print(f"Actual maximum: {actual_max:.2f}")
    #         print(f"\nThe data in this file has values higher than the expected")
    #         print(f"normalization threshold of {max_value}. This may indicate:")
    #         print("  - Incorrect reference file")
    #         print("  - Corrupted data")
    #         print("  - Different sensor configuration")
    #         print("\nREPORT THIS ERROR TO JUSTIN!")
    #         print("="*60)

    #         # Play system alert sound
    #         try:
    #             import subprocess
    #             # macOS system alert sound (works on Mac)
    #             subprocess.run(['afplay', '/System/Library/Sounds/Basso.aiff'], check=False)
    #         except Exception:
    #             # Fallback: try to use terminal bell
    #             print('\a')  # Terminal bell character

    #         # Show popup dialog
    #         try:
    #             import tkinter as tk
    #             from tkinter import messagebox

    #             # Create root window
    #             root = tk.Tk()
    #             root.withdraw()

    #             # Try to bring it to front on macOS
    #             root.lift()
    #             root.attributes('-topmost', True)
    #             root.after_idle(root.attributes, '-topmost', False)

    #             # Simple error message with warning at top only
    #             error_message = (
    #                 "🚨 REPORT THIS ERROR TO JUSTIN! 🚨\n\n"
    #                 f"File: {file_path}\n\n"
    #                 f"Expected maximum: {max_value}\n"
    #                 f"Actual maximum: {actual_max:.2f}\n\n"
    #                 f"This may indicate:\n"
    #                 f"  • Incorrect reference file\n"
    #                 f"  • Corrupted data\n"
    #                 f"  • Different sensor configuration"
    #             )

    #             messagebox.showerror(
    #                 "CRITICAL ERROR - REPORT TO JUSTIN!",
    #                 error_message
    #             )

    #             root.destroy()

    #         except Exception as e:
    #             print(f"Warning: Could not display popup dialog: {e}")

    #     return None

    # Normalize using fixed value of 290
    # measured = measured - 290

    # print ('MEASURED2 ')
    # print (measured.shape)
    # print (measured[39])

    # Cache the result before returning
    result = (measured, session_folder.name)
    _data_cache[cache_key] = result

    return result

def main():
    # Print help message at startup
    print_help()

    # Disable default matplotlib 's' key binding for saving figures
    # so we can use 's' for our own save regions function
    plt.rcParams['keymap.save'].remove('s')

    parser = argparse.ArgumentParser(
        description="Process session folder with multiple delta CSV files (one per frame)."
    )

    parser.add_argument("--f", help="Path to the session folder: data2/session_1325501210916,\
        if not specified all session folders are shown in GUI for selection")
    args = parser.parse_args()

    import glob
    import os

    # Search for session folders in current directory first, then data2/ subdirectories
    available_folders = []

    # Priority order: current dir, data2/, ../data2/
    search_paths = ['.', 'data2', '../data2']

    for search_path in search_paths:
        if os.path.exists(search_path):
            for item in os.listdir(search_path):
                # Only look for session_* folders
                if not item.startswith('session_'):
                    continue
                folder_path = os.path.join(search_path, item)
                if os.path.isdir(folder_path):
                    # Check if folder contains delta CSV files
                    delta_files = glob.glob(os.path.join(folder_path, "deltas_*.csv"))
                    if delta_files:
                        available_folders.append(folder_path)
        # If we found folders, stop searching other paths
        if available_folders:
            break

    if not available_folders:
        print("\n" + "="*60)
        print("ERROR: No maXTouch session folders found")
        print("="*60)
        print("\nThis script requires session folders with delta CSV files.")
        print("\nExpected folder structure:")
        print("  - data2/session_XXXXX/")
        print("    - deltas_0.csv")
        print("    - deltas_TIMESTAMP.csv")
        print("    - ...")
        print("  - Reference file: ref.csv (must exist in current directory)")
        print("\nSearched in:")
        print("  - Current directory for session_* folders")
        print("  - data2/ subdirectory")
        print("  - ../data2/ parent directory")
        print("="*60)
        sys.exit(1)

    # Sort folders with natural (human) sorting
    available_folders.sort(key=lambda x: natural_sort_key(os.path.basename(x)))

    if args.f:
        deltas = args.f
    else:
        # Use first folder by default
        deltas = available_folders[0]
        print(f"Multiple folders available. Starting with: {os.path.basename(deltas)}")

    # Load initial data file
    result = load_data_file(deltas)
    if result is None:
        sys.exit(1)

    measured_data, deltas_filename = result
    print(f"Data normalized: max value set to 0, all values now negative")
    print(f"Data range: {np.min(measured_data):.2f} to {np.max(measured_data):.2f}")

    # Store current file info and data
    current_file = [deltas]  # Use list for mutability in nested functions
    measured = measured_data
    T = measured.shape[0]

    print(measured.shape)

    # color scale - clamp to prevent extreme values from ruining the heatmap
    global vmin
    global vmax
    vmin = max(np.min(measured), -1000)  # Lower limit: -1000
    vmax = min(np.max(measured), 1000)   # Upper limit: +1000
    default_vmin = vmin
    default_vmax = vmax
    print(f"Color scale: {vmin:.2f} to {vmax:.2f} (clamped to [-1000, +1000])")

    # Auto-detect frame of interest (last significant change)
    def find_last_significant_change(data, filename, sensitivity=2.0, settling_frames=3):
        """
        Find the last frame where a significant change occurred (e.g., last drop placement).
        Uses different strategies based on drop size:
        - Small drops (10u, 25u, 50u): Select at the end of the last peak
        - Large drops (100u, 200u, 500u): Select after the signal settles post-peak

        Parameters:
        -----------
        data : np.ndarray
            3D array of shape (T, H, W) containing all frames
        filename : str
            Name of the file being processed (used to determine drop size)
        sensitivity : float
            Sensitivity multiplier for change detection. Lower = more sensitive.
            Default 2.0 means changes > mean + 2*std are considered significant.
        settling_frames : int
            Number of frames to look ahead after peak to find settling point.
            Default 3 frames.

        Returns:
        --------
        frame_idx : int
            Index of the frame after signal settles (large drops) or at peak end (small drops)
        diffs : np.ndarray
            Frame-to-frame difference values
        threshold : float
            Threshold value for significant changes
        """
        T = data.shape[0]

        if T < 2:
            return 0, np.array([]), 0

        # Determine drop size from filename
        import re
        is_small_drop = False
        is_100u = False
        drop_size_match = re.search(r'(\d+)u', filename.lower())
        if drop_size_match:
            drop_size = int(drop_size_match.group(1))
            is_small_drop = drop_size in [10, 25, 50]
            is_100u = drop_size == 100
            print(f"  Detected drop size: {drop_size}u ({'small' if is_small_drop else 'large'})")

        # Calculate frame-to-frame differences
        diffs = []
        for i in range(T - 1):
            diff = np.abs(data[i + 1] - data[i])
            # Use maximum difference as the metric (captures largest local changes)
            # This is more robust to noise and better captures drop placements
            max_diff = np.max(diff)
            diffs.append(max_diff)

        diffs = np.array(diffs)

        # Calculate threshold for "significant" change
        median_diff = np.median(diffs)
        std_diff = np.std(diffs)
        threshold = median_diff + sensitivity * std_diff

        print(f"\nChange detection statistics:")
        print(f"  Median frame difference: {median_diff:.2f}")
        print(f"  Std deviation: {std_diff:.2f}")
        print(f"  Significance threshold: {threshold:.2f}")

        # Find last significant peak
        # For small drops, we want the last single peak
        # For large drops, we want the second-to-last peak (to avoid the final rise)
        significant_frames = []
        for i in range(len(diffs) - 1, -1, -1):
            if diffs[i] > threshold:
                frame_idx = i + 1  # +1 because diff[i] is between frame i and i+1
                significant_frames.append((frame_idx, diffs[i]))
                if len(significant_frames) >= 2:
                    break

        if is_small_drop:
            # Small drops: select at the end of the last peak
            if len(significant_frames) >= 1:
                peak_frame = significant_frames[0][0]
                print(f"  Last significant change (peak): frame {peak_frame + 1}/{T} (magnitude: {significant_frames[0][1]:.2f})")

                # Find the end of this peak (when difference drops below threshold)
                end_of_peak = peak_frame
                for offset in range(1, min(10, T - peak_frame)):  # Look ahead up to 10 frames
                    check_idx = peak_frame + offset - 1
                    if check_idx < len(diffs) and diffs[check_idx] < threshold:
                        end_of_peak = peak_frame + offset
                        break
                else:
                    # If peak doesn't end, just use a small offset
                    end_of_peak = min(peak_frame + 2, T - 1)

                print(f"  End of peak at frame {end_of_peak + 1}/{T} (+{end_of_peak - peak_frame} frames after peak start)")
                return end_of_peak, diffs, threshold
            else:
                print(f"  No significant changes detected, starting at frame 1")
                return 0, diffs, threshold
        else:
            # Large drops: select a fixed number of frames after peak
            frames_after_peak = 5  # Fixed offset: select 5 frames after the peak

            if len(significant_frames) >= 2:
                # For 100u: use LAST peak (index 0)
                # For 200u/500u: use second-to-last peak (index 1)
                if is_100u:
                    peak_frame = significant_frames[0][0]
                    print(f"  Last significant change (peak): frame {peak_frame + 1}/{T} (magnitude: {significant_frames[0][1]:.2f})")
                else:
                    peak_frame = significant_frames[1][0]
                    print(f"  Last significant change: frame {significant_frames[0][0] + 1}/{T} (magnitude: {significant_frames[0][1]:.2f})")
                    print(f"  Second-to-last significant change (peak): frame {peak_frame + 1}/{T} (magnitude: {significant_frames[1][1]:.2f})")

                # Simply add fixed offset after peak
                settling_frame = min(peak_frame + frames_after_peak, T - 1)

                print(f"  Selected frame {settling_frame + 1}/{T} (+{settling_frame - peak_frame} frames after peak)")
                return settling_frame, diffs, threshold
            elif len(significant_frames) == 1:
                # Only one significant change found
                peak_frame = significant_frames[0][0]
                print(f"  Only one significant change detected at frame {peak_frame + 1}/{T}")
                print(f"  Change magnitude: {significant_frames[0][1]:.2f}")

                # Simply add fixed offset after peak
                settling_frame = min(peak_frame + frames_after_peak, T - 1)

                print(f"  Selected frame {settling_frame + 1}/{T} (+{settling_frame - peak_frame} frames after peak)")
                return settling_frame, diffs, threshold
            else:
                print(f"  No significant changes detected, starting at frame 1")
                return 0, diffs, threshold

    # Detect starting frame and get frame differences
    initial_frame, frame_diffs, diff_threshold = find_last_significant_change(measured, deltas_filename, sensitivity=2.0)

    # Function to detect frames with positive values
    def detect_positive_frames(data):
        """
        Detect which frames contain any positive pixel values.

        Parameters:
        -----------
        data : np.ndarray
            3D array of shape (T, H, W) containing all frames

        Returns:
        --------
        positive_frames : list
            List of frame indices that contain positive values
        """
        positive_frames = []
        T = data.shape[0]

        for frame_idx in range(T):
            frame = data[frame_idx]
            if np.any(frame > 0):
                positive_frames.append(frame_idx)

        return positive_frames

    # Detect frames with positive values
    positive_frames = detect_positive_frames(measured)
    print(f"Detected {len(positive_frames)} frames with positive values")

    # Initialize diff plot elements (will be populated later when plot is created)
    diff_plot_elements = [None, None, None, None]  # [line, threshold_line, current_frame_line, axes]
    positive_frame_markers = []  # Store markers for frames with positive values

    # State for region selection
    combined_mask = [None]  # Combined mask of all selected pixels
    region_overlay = [None]  # Visual overlay for selected region
    tolerance = [1.0]  # Tolerance for blob detection
    is_dragging_left = [False]  # Track if left mouse is being dragged
    is_dragging_right = [False]  # Track if right mouse is being dragged
    z_score_value = [2.0]  # Z-score threshold for auto-detection
    min_blob_size = [1]  # Minimum blob size in pixels
    frames_to_save = [1000]  # Number of frames to save (default 100, from current to end)
    patch_size = [6]  # Patch size for cropping (default 6x6)
    last_drag_position = [None]  # Track last position during drag to avoid redundant flood fills
    region_stack = [[]]  # Stack of individual region masks (each element is a region mask)
    regions_saved = [False]  # Track if current file's regions have been manually saved
    is_playing = [False]  # Track if animation is playing
    animation_timer = [None]  # Reference to the animation timer
    is_scrubbing_diff_plot = [False]  # Track if mouse is being dragged in diff plot for scrubbing
    include_frame_in_filename = [False]  # Whether to include frame number in saved filename (default: disabled)

    # figure with colorbar - make room for controls at bottom and file list on left
    fig = plt.figure(figsize=(20, 9))  # Increased height for better visibility
    # Main plot area - adjusted left margin to accommodate wider file list
    ax = fig.add_axes([0.22, 0.30, 0.58, 0.65])  # [left, bottom, width, height] - raised bottom from 0.25 to 0.30

    # Current frame index - start at detected frame of interest
    current_frame = [initial_frame]  # Use list to allow modification in nested function

    # Create custom colormap: green (negative) -> white (-250) -> red (positive)
    # Define colors at key points
    colors_list = ['green', 'white', 'red']
    n_bins = 256  # Number of discrete colors in the colormap
    custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom_diverging', colors_list, N=n_bins)

    # State variable to track current colormap
    current_cmap = ['custom']  # 'custom' or 'viridis'

    # Set the normalization so that 0 maps to the center (white)
    # We need to use TwoSlopeNorm to center the colormap at 0
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    # top image - set aspect='equal' so each pixel is square
    # This will make x-axis longer since W=52 > H=32
    im = ax.imshow(measured[initial_frame], norm=norm, aspect='equal', animated=True, cmap=custom_cmap)
    # Compute max value across all frames for display
    global_max = np.max(measured)
    title = ax.set_title(f"{os.path.basename(deltas)} — frame {initial_frame + 1}/{T} — max: {global_max:.1f}")

    # Set custom format_coord to display integer pixel coordinates
    ax.format_coord = lambda x, y: f'x={int(x+0.5)}, y={int(y+0.5)}' if 0 <= int(x+0.5) < measured.shape[2] and 0 <= int(y+0.5) < measured.shape[1] else ''

    # Add thin lightgrey gridlines at pixel boundaries
    H, W = measured[initial_frame].shape
    ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
    ax.grid(which='minor', color='lightgrey', linestyle='-', linewidth=0.5)
    ax.tick_params(which='minor', size=0)  # Hide minor tick marks

    # Create text annotations for displaying values in cells
    show_labels = [False]  # State variable to control label visibility (off by default)
    cell_value_texts = []
    H, W = measured[initial_frame].shape
    for y in range(H):
        row_texts = []
        for x in range(W):
            value = measured[initial_frame][y, x]
            color = 'green' if value > 0 else 'red'
            txt = ax.text(x, y, f'{int(value)}', ha='center', va='center',
                         fontsize=4, color=color, fontweight='bold', visible=show_labels[0])
            row_texts.append(txt)
        cell_value_texts.append(row_texts)

    # Add tolerance display text (top right area)
    tolerance_text = fig.text(0.89, 0.96, f'Tolerance: {tolerance[0]:.1f}',
                             fontsize=10,
                             verticalalignment='top',
                             horizontalalignment='right',
                             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Add z-score threshold display text (bottom right area)
    zscore_info_text = fig.text(0.87, 0.19, '',
                               fontsize=8,
                               verticalalignment='top',
                               horizontalalignment='center',
                               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    # Add timer display (top left above file list)
    start_time = time.time()
    timer_text = fig.text(0.11, 0.96, f'Time: 0:00',
                        fontsize=10,
                        verticalalignment='top',
                        horizontalalignment='center',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # Add cursor annotation (initially invisible)
    cursor_annotation = ax.annotate('', xy=(0, 0), xytext=(15, 15),
                                    textcoords='offset points',
                                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.9),
                                    fontsize=9,
                                    visible=False,
                                    zorder=1000,
                                    ha='center')

    # Add crosshairs (vertical and horizontal lines)
    crosshair_v = ax.axvline(x=0, color='red', linewidth=0.5, linestyle='--', alpha=0.7, visible=False, zorder=999)
    crosshair_h = ax.axhline(y=0, color='red', linewidth=0.5, linestyle='--', alpha=0.7, visible=False, zorder=999)

    # Track last cursor position to avoid redundant updates
    last_cursor_pos = [None]

    # Add colorbar with same height as the heatmap
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Value', rotation=270, labelpad=15)

    def update_frame(frame_idx):
        """Update the image and title to show the specified frame."""
        frame_idx = max(0, min(frame_idx, T - 1))  # Clamp to valid range
        current_frame[0] = frame_idx
        im.set_array(measured[frame_idx])
        # Compute max value across all frames for display
        global_max = np.max(measured)
        title.set_text(f"{os.path.basename(current_file[0])} — frame {frame_idx + 1}/{T} — max: {global_max:.1f}")

        # Update cell value texts
        H, W = measured[frame_idx].shape
        for y in range(H):
            for x in range(W):
                value = measured[frame_idx][y, x]
                cell_value_texts[y][x].set_text(f'{int(value)}')
                color = 'green' if value > 0 else 'red'
                cell_value_texts[y][x].set_color(color)

        # Update diff plot vertical line for current frame
        if diff_plot_elements[2] is not None:  # diff_current_frame_line
            diff_plot_elements[2].set_xdata([frame_idx, frame_idx])

        # Clear region selection when changing frames
        if region_overlay[0] is not None:
            try:
                if isinstance(region_overlay[0], list):
                    for artist in region_overlay[0]:
                        artist.remove()
                else:
                    region_overlay[0].remove()
            except:
                pass
            region_overlay[0] = None
        combined_mask[0] = None
        region_stack[0].clear()  # Clear the undo stack too

        # Run auto z-score detection on the new frame
        auto_detect_zscore(None)

        fig.canvas.draw_idle()

    def animate_frame():
        """Animation timer callback to advance to next frame."""
        if is_playing[0]:
            new_frame = current_frame[0] + 1
            if new_frame >= T:
                # Loop back to beginning
                new_frame = 0
            update_frame(new_frame)
        return True  # Keep timer running

    def toggle_play_pause(event):
        """Toggle animation play/pause state."""
        is_playing[0] = not is_playing[0]

        if is_playing[0]:
            # Start playing
            if animation_timer[0] is None:
                animation_timer[0] = fig.canvas.new_timer(interval=100)  # 10 fps
                animation_timer[0].add_callback(animate_frame)
            animation_timer[0].start()
            btn_play.label.set_text('Pause')
            print("Animation started")
        else:
            # Pause
            if animation_timer[0] is not None:
                animation_timer[0].stop()
            btn_play.label.set_text('Play')
            print("Animation paused")

        fig.canvas.draw_idle()

    def on_click(event):
        """Handle mouse clicks for region selection."""
        # Ignore clicks on scrollbar
        if event.inaxes == scrollbar_ax:
            return
        if event.inaxes != ax:
            return
        
        # Get click coordinates
        x_click, y_click = int(event.xdata + 0.5), int(event.ydata + 0.5)
        
        if event.button == 1:  # Left click - add to selection
            is_dragging_left[0] = True
            print(f"Left clicked at pixel: ({x_click}, {y_click})")

            # Perform flood fill
            current_data = measured[current_frame[0]]
            print(f"Data shape: {current_data.shape}, Value at click: {current_data[y_click, x_click]:.2f}")

            mask = flood_fill_region(current_data, y_click, x_click, tolerance[0])

            if mask is None or not mask.any():
                print("No region selected")
                return

            # Check if 6x6 patch around region contains any positive deviations (values > -280)
            y_coords, x_coords = np.where(mask)
            cy = int(np.mean(y_coords))
            cx = int(np.mean(x_coords))

            # Calculate 6x6 patch bounds
            crop_size = int(patch_size[0])
            half_size = crop_size // 2
            H, W = current_data.shape
            y_min = max(0, cy - half_size)
            y_max = min(H, cy + half_size)
            x_min = max(0, cx - half_size)
            x_max = min(W, cx + half_size)

            # Extract patch
            patch = current_data[y_min:y_max, x_min:x_max]

            # if np.any(patch > 0):
            #     max_val = patch.max()
            #     print(f"Region rejected: 6x6 patch contains positive values (max value: {max_val:.1f})")
            #     return

            # Save this region to the stack for undo functionality
            region_stack[0].append(mask.copy())

            # If we already have regions, add to them (union)
            if combined_mask[0] is not None:
                combined_mask[0] = combined_mask[0] | mask
            else:
                combined_mask[0] = mask

            update_overlay()
            
        elif event.button == 3:  # Right click - remove from selection
            is_dragging_right[0] = True
            print(f"Right clicked at pixel: ({x_click}, {y_click}) - deselecting")

            if combined_mask[0] is None:
                print("No selection to deselect from")
                return

            # Check if clicked pixel is actually selected
            if not combined_mask[0][y_click, x_click]:
                print("Clicked pixel is not selected")
                return

            # Find which region in the stack contains the clicked pixel
            region_to_remove = None
            region_idx_to_remove = None

            for idx, stacked_region in enumerate(region_stack[0]):
                # Check if the clicked pixel is in this stacked region
                if stacked_region[y_click, x_click]:
                    region_to_remove = stacked_region
                    region_idx_to_remove = idx
                    break

            if region_to_remove is None:
                print("No region found at clicked location in stack")
                return

            # Remove this region from the stack
            region_stack[0].pop(region_idx_to_remove)
            print(f"Removed region {region_idx_to_remove} from stack")

            # Remove the region from the combined mask
            combined_mask[0] = combined_mask[0] & ~region_to_remove

            # If mask is now empty, clear it completely
            if not combined_mask[0].any():
                combined_mask[0] = None
                if region_overlay[0] is not None:
                    try:
                        if isinstance(region_overlay[0], list):
                            for artist in region_overlay[0]:
                                artist.remove()
                        else:
                            region_overlay[0].remove()
                    except:
                        pass
                    region_overlay[0] = None
                fig.canvas.draw_idle()
                print("All regions deselected")
            else:
                update_overlay()

    def on_hover(event):
        """Update cursor annotation and crosshairs on mouse move - throttled updates."""
        if event.inaxes == ax and event.xdata is not None and event.ydata is not None:
            x_pos = int(event.xdata + 0.5)
            y_pos = int(event.ydata + 0.5)

            # Only update if position changed (avoid redundant updates)
            if last_cursor_pos[0] == (x_pos, y_pos):
                return
            last_cursor_pos[0] = (x_pos, y_pos)

            current_data = measured[current_frame[0]]

            # Check bounds
            if 0 <= y_pos < current_data.shape[0] and 0 <= x_pos < current_data.shape[1]:
                # Update cursor elements
                value = current_data[y_pos, x_pos]
                cursor_annotation.set_text(f'({x_pos:d}, {y_pos:d})\n{int(value)}')
                cursor_annotation.xy = (x_pos, y_pos)  # Snap to cell center
                cursor_annotation.set_visible(True)

                # Update crosshairs position - snap to cell centers
                crosshair_v.set_xdata([x_pos, x_pos])
                crosshair_h.set_ydata([y_pos, y_pos])
                crosshair_v.set_visible(True)
                crosshair_h.set_visible(True)

                # Request redraw
                fig.canvas.draw_idle()
            else:
                # Outside bounds - hide elements
                cursor_annotation.set_visible(False)
                crosshair_v.set_visible(False)
                crosshair_h.set_visible(False)
                last_cursor_pos[0] = None
                fig.canvas.draw_idle()
        else:
            # Outside axes - hide elements
            if cursor_annotation.get_visible():
                cursor_annotation.set_visible(False)
                crosshair_v.set_visible(False)
                crosshair_h.set_visible(False)
                last_cursor_pos[0] = None
                fig.canvas.draw_idle()

    # Track last processed position to avoid redundant flood fills
    last_drag_pos = [None]

    def on_motion(event):
        """Handle mouse motion for dragging to add/remove regions and scrubbing diff plot."""
        # Handle scrubbing on diff plot
        if is_scrubbing_diff_plot[0]:
            if event.inaxes == diff_plot_elements[3] and event.xdata is not None:
                frame_idx = int(round(event.xdata))
                # Clamp to valid range
                frame_idx = max(0, min(frame_idx, T - 1))
                # Update frame
                update_frame(frame_idx)
            return  # Don't handle region selection when scrubbing

        # Handle dragging for region selection/deselection
        if not (is_dragging_left[0] or is_dragging_right[0]):
            return
        if event.inaxes != ax:
            return

        # Get current coordinates
        x_click, y_click = int(event.xdata + 0.5), int(event.ydata + 0.5)

        # Skip if we're still on the same pixel (avoid redundant flood fills)
        if last_drag_pos[0] == (x_click, y_click):
            return
        last_drag_pos[0] = (x_click, y_click)

        # Perform flood fill at current position
        current_data = measured[current_frame[0]]
        mask = flood_fill_region(current_data, y_click, x_click, tolerance[0])

        if mask is None or not mask.any():
            return

        # Check if 6x6 patch around region contains any positive deviations (values > -280)
        y_coords, x_coords = np.where(mask)
        if len(y_coords) == 0:
            return
        cy = int(np.mean(y_coords))
        cx = int(np.mean(x_coords))

        # Calculate 6x6 patch bounds
        crop_size = int(patch_size[0])
        half_size = crop_size // 2
        H, W = current_data.shape
        y_min = max(0, cy - half_size)
        y_max = min(H, cy + half_size)
        x_min = max(0, cx - half_size)
        x_max = min(W, cx + half_size)

        # Extract patch
        patch = current_data[y_min:y_max, x_min:x_max]

        if np.any(patch > 0):
            return  # Skip regions with positive values in patch

        if is_dragging_left[0]:
            # Check if this region is not already selected (avoid duplicates)
            if combined_mask[0] is not None:
                # Only add if there are new pixels being selected
                if not (combined_mask[0] & mask).all() or not mask.all():
                    # Check if any new pixels will be added
                    new_pixels = mask & ~combined_mask[0]
                    if new_pixels.any():
                        # Save this region to the stack
                        region_stack[0].append(mask.copy())
                        combined_mask[0] = combined_mask[0] | mask
                        update_overlay()
            else:
                # First region being selected
                region_stack[0].append(mask.copy())
                combined_mask[0] = mask
                update_overlay()

        elif is_dragging_right[0]:
            # Remove from existing region
            if combined_mask[0] is not None:
                # Check if any pixel in the flood-filled mask overlaps with selected regions
                if not (combined_mask[0] & mask).any():
                    return  # No overlap, nothing to remove

                # Find which region(s) in the stack overlap with the dragged area
                # We'll remove the first one we find that has the clicked pixel
                region_to_remove = None
                region_idx_to_remove = None

                # Find all regions that overlap with the mask
                for idx, stacked_region in enumerate(region_stack[0]):
                    overlap = stacked_region & mask
                    if overlap.any():
                        # Remove this region
                        region_to_remove = stacked_region
                        region_idx_to_remove = idx
                        break

                if region_to_remove is not None:
                    region_stack[0].pop(region_idx_to_remove)
                    combined_mask[0] = combined_mask[0] & ~region_to_remove

                    # If mask is now empty, clear it completely
                    if not combined_mask[0].any():
                        combined_mask[0] = None
                        if region_overlay[0] is not None:
                            try:
                                if isinstance(region_overlay[0], list):
                                    for artist in region_overlay[0]:
                                        artist.remove()
                                else:
                                    region_overlay[0].remove()
                            except:
                                pass
                            region_overlay[0] = None
                        fig.canvas.draw_idle()
                    else:
                        update_overlay()

    def on_release(event):
        """Handle mouse release to stop dragging and scrubbing."""
        if event.button == 1:
            # Stop scrubbing if active
            if is_scrubbing_diff_plot[0]:
                is_scrubbing_diff_plot[0] = False
                print("Scrubbing complete")
            # Stop left drag if active
            elif is_dragging_left[0]:
                is_dragging_left[0] = False
                last_drag_pos[0] = None  # Reset position tracking
                print("Left drag complete")
        elif event.button == 3:
            is_dragging_right[0] = False
            last_drag_pos[0] = None  # Reset position tracking
            print("Right drag complete")

    def on_scroll(event):
        """Handle scroll wheel events to adjust tolerance (only when over heatmap)."""
        # Only adjust tolerance when mouse is over the heatmap
        if event.inaxes != ax:
            return

        if event.button == 'up':
            # Scroll up - increase tolerance
            tolerance[0] = min(tolerance[0] * 1.5, 1000.0)
            tolerance_text.set_text(f'Tolerance: {tolerance[0]:.1f}')
            fig.canvas.draw_idle()
            print(f"Tolerance increased to: {tolerance[0]:.2f}")
        elif event.button == 'down':
            # Scroll down - decrease tolerance
            tolerance[0] = max(tolerance[0] / 1.5, 0.1)
            tolerance_text.set_text(f'Tolerance: {tolerance[0]:.1f}')
            fig.canvas.draw_idle()
            print(f"Tolerance decreased to: {tolerance[0]:.2f}")

    def update_overlay():
        """Update the visual overlay without printing statistics."""
        if combined_mask[0] is None:
            return

        current_data = measured[current_frame[0]]

        # Remove old overlay
        if region_overlay[0] is not None:
            try:
                if isinstance(region_overlay[0], list):
                    for artist in region_overlay[0]:
                        artist.remove()
                else:
                    region_overlay[0].remove()
            except:
                pass

        # Draw red borders around selected pixels and patch boundaries
        from matplotlib.patches import Rectangle
        rectangles = []

        # Separate into individual regions (filtering out regions with positive values)
        individual_regions = separate_regions(combined_mask[0], current_data)

        # Create filtered mask containing only valid regions
        filtered_mask = np.zeros_like(combined_mask[0], dtype=bool)
        for region_mask in individual_regions:
            filtered_mask |= region_mask

        # Find all selected pixels (only from valid regions)
        y_coords, x_coords = np.where(filtered_mask)

        for y, x in zip(y_coords, x_coords):
            # Draw a rectangle border around this pixel
            # Rectangle position is at (x-0.5, y-0.5) with width and height of 1
            rect = Rectangle((x - 0.5, y - 0.5), 1, 1,
                           linewidth=0.5, edgecolor='red', facecolor='none', zorder=10)
            ax.add_patch(rect)
            rectangles.append(rect)

        # Draw patch boundaries around each region
        crop_size = int(patch_size[0])

        for region_mask in individual_regions:
            # Calculate region stats to get centroid
            from region_stats import calculate_region_stats
            stats = calculate_region_stats(region_mask, current_data, pixel_spacing_mm)
            cy, cx = stats['centroid_y'], stats['centroid_x']

            # Get region characteristics for adaptive centering
            pixel_count = stats['pixel_count']
            region_height = stats['height_pixels']
            region_width = stats['width_pixels']

            # Choose offset strategy (same as in save_regions)
            if pixel_count <= 2 and (region_height <= 2 or region_width <= 2):
                offset_before = crop_size // 2 - 1 if crop_size > 2 else 0
            else:
                offset_before = crop_size // 2

            # Draw the patch boundary
            # Rectangle position is centered on the centroid
            patch_x = cx - offset_before - 0.5
            patch_y = cy - offset_before - 0.5

            patch_rect = Rectangle((patch_x, patch_y), crop_size, crop_size,
                                  linewidth=2, edgecolor='lime', facecolor='none',
                                  linestyle='--', alpha=0.8, zorder=11)
            ax.add_patch(patch_rect)
            rectangles.append(patch_rect)

        region_overlay[0] = rectangles

        fig.canvas.draw_idle()

    def print_statistics(event=None):
        """Print detailed statistics for currently selected regions."""
        if combined_mask[0] is None:
            print("\nNo regions selected.")
            return

        current_data = measured[current_frame[0]]

        # Separate into individual connected regions (filtering out regions with positive values)
        individual_regions = separate_regions(combined_mask[0], current_data)
        num_regions = len(individual_regions)

        print(f"\n{'='*60}")
        print(f"Frame: {current_frame[0] + 1}/{T}")
        print(f"Number of separate regions detected: {num_regions}")
        print(f"{'='*60}")

        # Calculate and print statistics for each region using module functions
        if num_regions > 0:
            for idx, region_mask in enumerate(individual_regions, 1):
                stats = calculate_region_stats(region_mask, current_data, pixel_spacing_mm)
                print_region_stats(stats, idx, num_regions)

            # Calculate and print summary statistics using module functions
            summary = calculate_summary_stats(individual_regions, combined_mask[0],
                                             current_data, pixel_spacing_mm)
            print_summary_stats(summary)

            print(f"\nTolerance: {tolerance[0]:.2f}")
            print(f"{'='*60}\n")

    def on_key(event):
        """Handle keyboard events."""
        global vmin
        global vmax
        if event.key == ' ':
            # Toggle play/pause with space bar
            toggle_play_pause(None)
        elif event.key == '0':
            # Jump to frame 0
            update_frame(0)
        elif event.key == '1':
            # Jump to frame 1
            update_frame(1)
        elif event.key == 'd':
            # Advance frame
            new_frame = current_frame[0] + 1
            update_frame(new_frame)
        elif event.key == 'a':
            # Go back frame
            new_frame = current_frame[0] - 1
            update_frame(new_frame)
        elif event.key == 'up':
            # Move up in file list (previous file)
            new_idx = current_file_idx[0] - 1
            if new_idx >= 0:
                load_file_by_index(new_idx)
            else:
                print("Already at first file")
        elif event.key == 'down':
            # Move down in folder list (next folder)
            new_idx = current_file_idx[0] + 1
            if new_idx < len(available_folders):
                load_file_by_index(new_idx)
            else:
                print("Already at last folder")
        elif event.key == 'u':
            # Undo last region selection
            undo_region(None)
        elif event.key == 'z':
            # Decrease vmin
            if vmin-50 < vmax:
                vmin -= 50
                new_norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
                im.set_norm(new_norm)
                print(f"vmin decreased to: {vmin}")
                fig.canvas.draw_idle()
        elif event.key == 'c':
            # Increase vmin
            if vmin+50<vmax:
                vmin += 50
                new_norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
                im.set_norm(new_norm)
                print(f"vmin increased to: {vmin}")
                fig.canvas.draw_idle()
        elif event.key == ',':
            # Decrease vmax
            if vmax-50 > vmin:
                vmax -= 50
                new_norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
                im.set_norm(new_norm)
                print(f"vmax decreased to: {vmax}")
                fig.canvas.draw_idle()
        elif event.key == '/':
            # Increase vmax
            if vmax+50>vmin:
                vmax += 50
                new_norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
                im.set_norm(new_norm)
                print(f"vmax increased to: {vmax}")
                fig.canvas.draw_idle()
        elif event.key in ['+', '=']:
            # Increase tolerance (both + and = keys for convenience)
            tolerance[0] = min(tolerance[0] * 1.5, 1000.0)
            tolerance_text.set_text(f'Tolerance: {tolerance[0]:.1f}')
            fig.canvas.draw_idle()
            print(f"Tolerance increased to: {tolerance[0]:.2f}")
        elif event.key in ['-', '_']:
            # Decrease tolerance (both - and _ keys for convenience)
            tolerance[0] = max(tolerance[0] / 1.5, 0.1)
            tolerance_text.set_text(f'Tolerance: {tolerance[0]:.1f}')
            fig.canvas.draw_idle()
            print(f"Tolerance decreased to: {tolerance[0]:.2f}")
        elif event.key == 'x':
            # Clear selection
            if region_overlay[0] is not None:
                try:
                    if isinstance(region_overlay[0], list):
                        for artist in region_overlay[0]:
                            artist.remove()
                    else:
                        region_overlay[0].remove()
                except:
                    pass
                region_overlay[0] = None
            combined_mask[0] = None
            region_stack[0].clear()  # Clear the undo stack too
            fig.canvas.draw_idle()
            print("Selection cleared")
        elif event.key == 's':
            # Save regions
            save_regions()
        elif event.key == 'p':
            # Print statistics
            print_statistics()

    def save_regions(event=None):
        """Save regions to disk for current frame + N consecutive frames."""
        if combined_mask[0] is None:
            print("\nNo regions to save.")
            return

        current_data = measured[current_frame[0]]

        # Separate into individual connected regions (filtering out regions with positive values)
        individual_regions = separate_regions(combined_mask[0], current_data)
        num_regions = len(individual_regions)

        if num_regions == 0:
            print("\nNo regions to save.")
            return

        # Determine frame range (current frame to end of data)
        start_frame = current_frame[0]
        end_frame = T  # Save all frames from current to end
        num_frames = end_frame - start_frame
        frame_indices = list(range(start_frame, end_frame))

        print(f"\nSaving regions:")
        print(f"  Current frame: {start_frame + 1} of {T}")
        print(f"  Frames remaining (including current): {T - start_frame}")
        print(f"  Will save frames {start_frame + 1} to {end_frame} ({num_frames} frames total)")

        # Create regions directory if it doesn't exist
        regions_dir = Path("regions")
        regions_dir.mkdir(parents=True, exist_ok=True)

        # Create output filename based on input file and optionally include frame number
        if include_frame_in_filename[0]:
            output_file = regions_dir / (Path(current_file[0]).stem + f"_frame{start_frame}_regions.npz")
        else:
            output_file = regions_dir / (Path(current_file[0]).stem + "_regions.npz")

        # Prepare data to save
        region_data = {
            'combined_mask': combined_mask[0],
            'num_regions': num_regions,
            'tolerance': tolerance[0],
            'source_file': str(current_file[0]),
            'start_frame': start_frame,
            'num_frames_captured': num_frames,
            'frame_indices': np.array(frame_indices)
        }

        # Save individual region masks, statistics, and crops for all frames
        crop_size = int(patch_size[0])
        for idx, region_mask in enumerate(individual_regions):
            y_coords, x_coords = np.where(region_mask)

            region_data[f'region_{idx}_mask'] = region_mask
            region_data[f'region_{idx}_coords'] = np.column_stack((y_coords, x_coords))

            # Calculate centroid for cropping
            stats_first_frame = calculate_region_stats(region_mask, measured[start_frame], pixel_spacing_mm)
            cy, cx = stats_first_frame['centroid_y'], stats_first_frame['centroid_x']
            half_size = crop_size // 2

            # Calculate crop bounds with boundary handling
            # For even-sized crops (like 6x6), we need to choose the centering strategy
            # based on the region characteristics
            H, W = measured.shape[1], measured.shape[2]

            # Get region pixel count and dimensions
            pixel_count = stats_first_frame['pixel_count']
            region_height = stats_first_frame['height_pixels']
            region_width = stats_first_frame['width_pixels']

            # Choose offset strategy:
            # - For 1-2 pixel regions (especially linear): use offset that puts centroid at index 2
            #   This gives more space after the region (2 before, 3 after)
            # - For larger regions: use symmetric offset (3 before, 3 after)
            if pixel_count <= 2 and (region_height <= 2 or region_width <= 2):
                # Small linear region: offset centroid towards top-left
                offset_before = crop_size // 2 - 1  # For size=6, this is 2
                offset_after = crop_size - offset_before  # For size=6, this is 4
            else:
                # Larger region: center symmetrically
                offset_before = crop_size // 2  # For size=6, this is 3
                offset_after = crop_size // 2  # For size=6, this is 3

            # First calculate ideal bounds (may exceed image boundaries)
            y_min = cy - offset_before
            y_max = cy + offset_after
            x_min = cx - offset_before
            x_max = cx + offset_after

            # Apply boundary constraints
            y_min = max(0, y_min)
            x_min = max(0, x_min)
            y_max = min(H, y_max)
            x_max = min(W, x_max)

            # Extract crops for all frames
            all_frames_crops = []
            for frame_idx in frame_indices:
                frame_data = measured[frame_idx]

                # Extract crop
                crop = frame_data[y_min:y_max, x_min:x_max]

                # Pad if near boundary
                if crop.shape != (crop_size, crop_size):
                    padded = np.zeros((crop_size, crop_size), dtype=crop.dtype)
                    # Calculate offset: how much padding is needed at the start?
                    # Ideal range is [cy-offset_before : cy+offset_after]
                    # Actual range is [y_min : y_max] after boundary clipping
                    # Padding needed at start = how many pixels we couldn't get from the left/top
                    y_offset = max(0, offset_before - cy)  # Padding at top
                    x_offset = max(0, offset_before - cx)  # Padding at left

                    # Copy the crop into the padded array at the correct offset
                    padded[y_offset:y_offset+crop.shape[0], x_offset:x_offset+crop.shape[1]] = crop
                    crop = padded

                all_frames_crops.append(crop)

                # Calculate and save statistics for this frame
                stats = calculate_region_stats(region_mask, frame_data, pixel_spacing_mm)
                region_data[f'region_{idx}_frame_{frame_idx}_stats'] = stats

            # Stack crops into single array (num_frames, crop_size, crop_size)
            region_data[f'region_{idx}_crops'] = np.stack(all_frames_crops, axis=0)

            # Create crop mask (which pixels in the crop are part of the region)
            crop_region_mask = region_mask[y_min:y_max, x_min:x_max]

            # Pad mask if necessary
            if crop_region_mask.shape != (crop_size, crop_size):
                padded_mask = np.zeros((crop_size, crop_size), dtype=crop_region_mask.dtype)
                # Calculate offset using same logic as crop padding
                y_offset = max(0, offset_before - cy)
                x_offset = max(0, offset_before - cx)

                # Copy the mask into the padded array at the correct offset
                padded_mask[y_offset:y_offset+crop_region_mask.shape[0],
                           x_offset:x_offset+crop_region_mask.shape[1]] = crop_region_mask
                crop_region_mask = padded_mask

            region_data[f'region_{idx}_crop_mask'] = crop_region_mask
            region_data[f'region_{idx}_crop_centroid'] = (cy, cx)
            region_data[f'region_{idx}_crop_bounds'] = (y_min, y_max, x_min, x_max)

        # DEBUG: Print region matrices before saving
        print(f"\n{'='*60}")
        print(f"DEBUG: REGION MATRICES BEING SAVED")
        print(f"{'='*60}")

        for idx, region_mask in enumerate(individual_regions):
            # Get the bounding box for this region
            y_coords, x_coords = np.where(region_mask)
            y_min, y_max = y_coords.min(), y_coords.max()
            x_min, x_max = x_coords.min(), x_coords.max()

            # Extract region data from first frame
            frame_data = measured[start_frame]
            region_bbox_data = frame_data[y_min:y_max+1, x_min:x_max+1].copy()
            region_bbox_mask = region_mask[y_min:y_max+1, x_min:x_max+1]

            # Set non-region pixels to NaN for visualization
            region_bbox_data_masked = region_bbox_data.astype(float)
            region_bbox_data_masked[region_bbox_mask == 0] = np.nan

            print(f"\n--- Region {idx} (Frame {start_frame + 1}) ---")
            print(f"Bounding box: {y_max-y_min+1} x {x_max-x_min+1} pixels")
            print(f"Region pixels shown with values, non-region pixels shown as 'nan':\n")

            # Print the matrix
            np.set_printoptions(precision=2, suppress=True, linewidth=200, nanstr='   nan')
            print(region_bbox_data_masked)
            np.set_printoptions()  # Reset to defaults

            # Print the mask for verification
            print(f"\nRegion mask (1=in region, 0=not in region):")
            print(region_bbox_mask.astype(int))

            # Print the crop and its mask
            print(f"\n{crop_size}x{crop_size} Crop (centered on centroid {region_data[f'region_{idx}_crop_centroid']}):")
            crop_first_frame = region_data[f'region_{idx}_crops'][0]  # First frame
            np.set_printoptions(precision=2, suppress=True, linewidth=200)
            print(crop_first_frame)
            np.set_printoptions()  # Reset to defaults

            print(f"\n{crop_size}x{crop_size} Crop mask (1=in region, 0=not in region or padding):")
            crop_mask = region_data[f'region_{idx}_crop_mask']
            print(crop_mask.astype(int))

            print(f"\nCrop bounds in original image: y=[{region_data[f'region_{idx}_crop_bounds'][0]}:{region_data[f'region_{idx}_crop_bounds'][1]}], x=[{region_data[f'region_{idx}_crop_bounds'][2]}:{region_data[f'region_{idx}_crop_bounds'][3]}]")
            print(f"Crop shape saved: {region_data[f'region_{idx}_crops'].shape} (num_frames x {crop_size} x {crop_size})")

        print(f"\n{'='*60}\n")

        # Save to disk
        np.savez(output_file, **region_data)

        # Create text file with centroid coordinates and amplitudes
        # txt_output_file = regions_dir / (Path(current_file[0]).stem + "_centroids.txt")
        # with txt_output_file.open('w') as f:
        #     f.write("# Region centroids and amplitudes\n")
        #     f.write("# Format: region_id, x, y, amplitude (min value)\n")
        #     f.write("#\n")

        #     for idx in range(num_regions):
        #         # Get stats from the first frame
        #         stats = region_data[f'region_{idx}_frame_{start_frame}_stats']
        #         centroid_x = stats['centroid_x']
        #         centroid_y = stats['centroid_y']
        #         amplitude = stats['min']  # Use min value as amplitude

        #         f.write(f"{idx}, {centroid_x}, {centroid_y}, {amplitude:.4f}\n")

        print(f"{'='*60 }")
        print(f"Saved {num_regions} region(s) to: {output_file}")
        # print(f"Saved centroids to: {txt_output_file}")
        print(f"Frames: {start_frame + 1} to {end_frame} ({num_frames} frames total)")
        print(f"Tolerance used: {tolerance[0]:.2f}")
        print(f"{'='*60}\n")

        # Print centroid amplitude values for easy copying
        print("Centroid amplitude values:")
        for idx in range(num_regions):
            # Get stats from the first frame
            stats = region_data[f'region_{idx}_frame_{start_frame}_stats']
            amplitude = stats['min']  # Use min value as amplitude
            print(f"{amplitude:.4f}")

        # Mark that regions have been saved for this file
        regions_saved[0] = True

    def on_close(event):
        """Print elapsed time, log to file, and automatically save regions when window is closed."""
        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)

        # Print to console
        print(f"\n{'='*60}")
        print(f"Session ended")
        print(f"Total time: {minutes}:{seconds:02d} ({elapsed_time:.1f} seconds)")
        print(f"{'='*60}\n")

        # Check if regions were selected
        has_regions = combined_mask[0] is not None and combined_mask[0].any()

        if not has_regions:
            print("No regions selected - session not logged or saved")
            return

        # Note: Automatic saving disabled - use 's' key or Save Regions button to save manually
        print("\nNote: Auto-save disabled. Use 's' key or 'Save Regions' button to save manually.")

        # Log to file
        log_file = Path("labeling_log.csv")
        file_exists = log_file.exists()

        # Get current timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Read existing entries and filter out previous entry for this file
        existing_lines = []
        if file_exists:
            with log_file.open('r') as f:
                lines = f.readlines()
                # Keep header and all entries except those for the current file
                if len(lines) > 0:
                    existing_lines.append(lines[0])  # Keep header
                    for line in lines[1:]:
                        # Parse CSV line to get filename
                        parts = line.strip().split(',')
                        if len(parts) >= 2 and parts[1] != os.path.basename(current_file[0]):
                            existing_lines.append(line)

        # Write back all entries plus the new one
        with log_file.open('w') as f:
            # Write header if new file or restore existing header
            if not file_exists:
                f.write("timestamp,csv_file,duration_seconds,duration_formatted\n")
            else:
                if existing_lines:
                    for line in existing_lines:
                        f.write(line)

            # Write new log entry
            f.write(f"{timestamp},{os.path.basename(current_file[0])},{elapsed_time:.1f},{minutes}:{seconds:02d}\n")

        print(f"Session logged to: {log_file}")

    # Add auto-detection buttons and sliders
    def auto_detect_zscore(event):
        """Auto-detect blobs using Z-score thresholding (negative direction only).
        Extreme values outside [-1000, +1000] are treated as 0."""
        current_data = measured[current_frame[0]]
        
        # Clamp extreme values to prevent abnormal values from affecting detection
        current_data_clamped = clamp_extreme_values(current_data)

        # Calculate mean and std for display (from clamped data)
        mean = np.mean(current_data_clamped)
        std = np.std(current_data_clamped)

        # Use negative threshold only (mean - z×std)
        threshold = mean - z_score_value[0] * std
        mask = current_data_clamped < threshold

        # Remove small blobs using 8-connectivity
        structure = np.array([[1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1]], dtype=bool)
        from scipy import ndimage
        labeled, num_features = ndimage.label(mask, structure=structure)
        for i in range(1, num_features + 1):
            region_mask = (labeled == i)
            if np.sum(region_mask) < int(min_blob_size[0]):
                mask[region_mask] = False

        # Update z-score info display
        zscore_info_text.set_text(
            f'Z-score: {z_score_value[0]:.1f} | '
            f'Mean: {mean:.1f} | Std: {std:.1f} | '
            f'Threshold: {threshold:.1f}'
        )
        fig.canvas.draw_idle()

        if mask.any():
            # Separate into individual regions and add each to the stack (filtering out regions with positive values)
            individual_regions = separate_regions(mask, current_data)

            # Filter out regions where length and width differ by more than 2 pixels
            filtered_regions = []
            excluded_count = 0
            for region_mask in individual_regions:
                # Calculate region dimensions
                y_coords, x_coords = np.where(region_mask)
                height = y_coords.max() - y_coords.min() + 1
                width = x_coords.max() - x_coords.min() + 1

                # Check if dimensions are within tolerance (differ by at most 2 pixels)
                if abs(height - width) <= 2:
                    filtered_regions.append(region_mask)
                    region_stack[0].append(region_mask.copy())
                else:
                    excluded_count += 1

            if filtered_regions:
                # Combine filtered regions into mask
                combined_mask[0] = np.zeros_like(mask)
                for region_mask in filtered_regions:
                    combined_mask[0] = combined_mask[0] | region_mask

                update_overlay()
                print(f"Z-score auto-detection: threshold = {threshold:.2f} "
                      f"(z={z_score_value[0]:.1f}), {np.sum(combined_mask[0])} pixels selected in {len(filtered_regions)} regions")
                if excluded_count > 0:
                    print(f"  Excluded {excluded_count} region(s) with length/width difference > 2 pixels")
            else:
                print(f"No blobs detected with Z-score = {z_score_value[0]:.1f} after filtering")
        else:
            print(f"No blobs detected with Z-score = {z_score_value[0]:.1f}")

    def undo_region(event):
        """Undo the most recently added region selection."""
        if len(region_stack[0]) == 0:
            print("No regions to undo")
            return

        # Pop the most recent region from the stack
        last_region = region_stack[0].pop()

        # Remove this region from the combined mask
        if combined_mask[0] is not None:
            combined_mask[0] = combined_mask[0] & ~last_region

            # If mask is now empty, clear it completely
            if not combined_mask[0].any():
                combined_mask[0] = None
                if region_overlay[0] is not None:
                    try:
                        if isinstance(region_overlay[0], list):
                            for artist in region_overlay[0]:
                                artist.remove()
                        else:
                            region_overlay[0].remove()
                    except:
                        pass
                    region_overlay[0] = None
                fig.canvas.draw_idle()
                print("Undid last region - all regions now cleared")
            else:
                update_overlay()
                print("Undid last region selection")
        else:
            print("No combined mask to remove from")

    def reset_colormap(event):
        """Reset color scale to default values."""
        global vmin, vmax
        vmin = default_vmin
        vmax = default_vmax
        # Update the norm with new vmin/vmax while keeping vcenter at 0
        new_norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        im.set_norm(new_norm)
        fig.canvas.draw_idle()
        print(f"Color scale reset to default: vmin={vmin:.2f}, vmax={vmax:.2f}")

    def toggle_colormap(event):
        """Toggle between custom and viridis colormap."""
        if current_cmap[0] == 'custom':
            im.set_cmap('viridis')
            current_cmap[0] = 'viridis'
            btn_toggle_cmap.label.set_text('Custom')
            print("Switched to viridis colormap")
        else:
            im.set_cmap(custom_cmap)
            current_cmap[0] = 'custom'
            btn_toggle_cmap.label.set_text('Viridis')
            print("Switched to custom colormap (green-white-red)")
        fig.canvas.draw_idle()

    def update_zscore(val):
        """Update z-score value from slider."""
        z_score_value[0] = val
        print(f"Z-score threshold updated to: {val:.2f}")

        # Update display with new threshold calculation (negative direction only)
        # Use clamped data for accurate display
        current_data = measured[current_frame[0]]
        current_data_clamped = clamp_extreme_values(current_data)
        mean = np.mean(current_data_clamped)
        std = np.std(current_data_clamped)
        threshold = mean - val * std

        zscore_info_text.set_text(
            f'Z-score: {val:.1f} | '
            f'Mean: {mean:.1f} | Std: {std:.1f} | '
            f'Threshold: {threshold:.1f}'
        )
        fig.canvas.draw_idle()
        fig.canvas.draw_idle()

    def update_minsize(val):
        """Update minimum blob size from slider."""
        min_blob_size[0] = int(val)
        print(f"Minimum blob size updated to: {int(val)} pixels")

    def update_patchsize(val):
        """Update patch size from slider."""
        patch_size[0] = int(val)
        print(f"Patch size updated to: {int(val)}x{int(val)}")
        # Update the overlay to show new patch size
        if combined_mask[0] is not None:
            update_overlay()

    # Create Play/Pause button
    btn_play_ax = fig.add_axes([0.08, 0.11, 0.08, 0.04])
    btn_play = Button(btn_play_ax, 'Play')
    btn_play.on_clicked(toggle_play_pause)

    # Create button for Z-score auto-detection
    btn_zscore_ax = fig.add_axes([0.20, 0.11, 0.12, 0.04])
    btn_zscore = Button(btn_zscore_ax, 'Auto: Z-score')
    btn_zscore.on_clicked(auto_detect_zscore)

    # Create button for resetting colormap to default
    btn_reset_colormap_ax = fig.add_axes([0.35, 0.11, 0.10, 0.04])
    btn_reset_colormap = Button(btn_reset_colormap_ax, 'Reset CM')
    btn_reset_colormap.on_clicked(reset_colormap)

    # Create button for toggling colormap
    btn_toggle_cmap_ax = fig.add_axes([0.46, 0.11, 0.10, 0.04])
    btn_toggle_cmap = Button(btn_toggle_cmap_ax, 'Viridis')
    btn_toggle_cmap.on_clicked(toggle_colormap)

    # Create button for printing statistics
    btn_print_stats_ax = fig.add_axes([0.57, 0.11, 0.10, 0.04])
    btn_print_stats = Button(btn_print_stats_ax, 'Print Stats')
    btn_print_stats.on_clicked(print_statistics)

    # Create button for saving regions
    btn_save_regions_ax = fig.add_axes([0.68, 0.11, 0.10, 0.04])
    btn_save_regions = Button(btn_save_regions_ax, 'Save Regions')
    btn_save_regions.on_clicked(save_regions)

    # Create undo button
    btn_undo_ax = fig.add_axes([0.79, 0.11, 0.06, 0.04])
    btn_undo = Button(btn_undo_ax, 'Undo')
    btn_undo.on_clicked(undo_region)

    # Create checkbox for showing cell labels
    def toggle_labels(label):
        """Toggle visibility of cell value labels."""
        show_labels[0] = not show_labels[0]
        for y in range(len(cell_value_texts)):
            for x in range(len(cell_value_texts[y])):
                cell_value_texts[y][x].set_visible(show_labels[0])
        fig.canvas.draw_idle()
        print(f"Cell labels: {'ON' if show_labels[0] else 'OFF'}")

    checkbox_labels_ax = fig.add_axes([0.82, 0.11, 0.10, 0.04])
    checkbox_labels = CheckButtons(checkbox_labels_ax, ['Show cell values'], [show_labels[0]])
    checkbox_labels.on_clicked(toggle_labels)

    # Create checkbox for including frame number in filename
    def toggle_frame_in_filename(label):
        """Toggle whether to include frame number in saved filename."""
        include_frame_in_filename[0] = not include_frame_in_filename[0]
        status = 'ON' if include_frame_in_filename[0] else 'OFF'
        print(f"Include frame in filename: {status}")
        print(f"  Filename format: session_name{'_frameN' if include_frame_in_filename[0] else ''}_regions.npz")

    checkbox_frame_filename_ax = fig.add_axes([0.82, 0.06, 0.10, 0.04])
    checkbox_frame_filename = CheckButtons(checkbox_frame_filename_ax, ['Frame in filename'], [include_frame_in_filename[0]])
    checkbox_frame_filename.on_clicked(toggle_frame_in_filename)

    # Create slider for Z-score threshold
    slider_zscore_ax = fig.add_axes([0.20, 0.04, 0.40, 0.02])
    slider_zscore = Slider(slider_zscore_ax, 'Z-score', 0.1, 5.0,
                           valinit=z_score_value[0], valstep=0.1)
    slider_zscore.on_changed(update_zscore)

    # Create slider for patch size
    slider_patchsize_ax = fig.add_axes([0.20, 0.01, 0.40, 0.02])
    slider_patchsize = Slider(slider_patchsize_ax, 'Patch Size', 2, 20,
                              valinit=patch_size[0], valstep=1)
    slider_patchsize.on_changed(update_patchsize)

    # Add save info text between buttons and slider (moved up slightly)
    fig.text(0.50, 0.085, "Press 's' to save regions (auto-save disabled - manual save only)",
             fontsize=8, style='italic', color='darkgreen',
             horizontalalignment='center',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.3))

    # Create axes for frame difference plot
    diff_plot_ax = fig.add_axes([0.22, 0.20, 0.58, 0.08])
    # X-axis should match frame indices (0 to T-1), diff values are plotted at frame i (representing diff from i to i+1)
    diff_plot_ax.set_xlim(0, T - 1)
    # Set y-axis to show variation better - tight range to emphasize changes
    if len(frame_diffs) > 0:
        # Use min and max with some padding to show all data clearly
        y_min = np.min(frame_diffs)
        y_max = np.max(frame_diffs)
        # Add padding (10% of range) for visual clarity
        y_range = y_max - y_min
        diff_plot_ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
    else:
        diff_plot_ax.set_ylim(0, 1)
    diff_plot_ax.set_ylabel('Change', fontsize=8)
    diff_plot_ax.set_xlabel('Frame', fontsize=8)
    diff_plot_ax.tick_params(axis='both', which='major', labelsize=7)

    # Plot the frame differences
    # Note: frame_diffs[i] represents the difference between frame i and i+1
    # We plot at x position i (same as frame index i)
    if len(frame_diffs) > 0:
        diff_line, = diff_plot_ax.plot(range(len(frame_diffs)), frame_diffs,
                                        linewidth=1, color='blue', alpha=0.7)
        # Add threshold line
        diff_threshold_line = diff_plot_ax.axhline(y=diff_threshold, color='red',
                                                    linestyle='--', linewidth=0.5, alpha=0.7)
        # Add vertical line for current frame
        diff_current_frame_line = diff_plot_ax.axvline(x=current_frame[0], color='green',
                                                        linewidth=1.5, alpha=0.8)
        # Store references for updates
        diff_plot_elements[0] = diff_line
        diff_plot_elements[1] = diff_threshold_line
        diff_plot_elements[2] = diff_current_frame_line
        diff_plot_elements[3] = diff_plot_ax

        # Add green markers for frames with positive values
        # Draw short vertical lines at the top of the plot
        y_min, y_max = diff_plot_ax.get_ylim()
        marker_height = (y_max - y_min) * 0.15  # 15% of plot height
        marker_y_top = y_max
        marker_y_bottom = y_max - marker_height

        for frame_idx in positive_frames:
            if frame_idx < T:  # Ensure frame is within bounds
                marker_line = diff_plot_ax.plot([frame_idx, frame_idx],
                                                 [marker_y_bottom, marker_y_top],
                                                 color='lime', linewidth=2, alpha=0.6,
                                                 zorder=5)[0]  # zorder=5 to appear above threshold line
                positive_frame_markers.append(marker_line)


    # File selector panel on the left - split into file list and scrollbar
    file_list_ax = fig.add_axes([0.02, 0.30, 0.17, 0.65])  # Slightly narrower to make room for scrollbar
    file_list_ax.set_title('Files (click to select)', fontsize=9, pad=5)
    file_list_ax.set_xlim(0, 1)

    # Scrollbar axes (thin strip on the right side of file list)
    scrollbar_ax = fig.add_axes([0.19, 0.30, 0.01, 0.65])
    scrollbar_ax.set_xlim(0, 1)
    scrollbar_ax.set_ylim(0, 1)
    scrollbar_ax.axis('off')

    # Scrollable file list parameters
    max_visible_files = 30  # Maximum number of files visible at once
    file_scroll_offset = [0]  # Current scroll offset (which file is at the top) - integer for instant scrolling

    # Set y-limits to show only max_visible_files at a time
    file_list_ax.set_ylim(0, max_visible_files)
    file_list_ax.axis('off')

    # Create file list text items
    file_text_items = []
    file_rects = []
    scrollbar_rect = [None]  # Scrollbar indicator
    scrollbar_track_rect = [None]  # Scrollbar track background
    is_dragging_scrollbar = [False]  # Track if scrollbar is being dragged
    current_file_idx = [available_folders.index(current_file[0])]

    def update_file_list_display():
        """Update the file list display based on scroll offset."""
        # Clear existing items
        for rect in file_rects:
            rect.remove()
        for txt in file_text_items:
            txt.remove()
        file_rects.clear()
        file_text_items.clear()

        # Calculate visible range
        start_idx = file_scroll_offset[0]
        end_idx = min(start_idx + max_visible_files, len(available_folders))

        # Draw visible files
        for idx, i in enumerate(range(start_idx, end_idx)):
            file_path = available_folders[i]
            # Position files from top to bottom
            y_pos = max_visible_files - idx - 0.5

            # Highlight current file
            if i == current_file_idx[0]:
                bgcolor = 'lightblue'
                fontweight = 'bold'
            else:
                bgcolor = 'white'
                fontweight = 'normal'

            # Background rectangle for click detection
            from matplotlib.patches import Rectangle
            rect = Rectangle((0, y_pos - 0.4), 0.98, 0.8,
                            facecolor=bgcolor, edgecolor='gray', linewidth=0.5,
                            picker=True, gid=str(i))
            file_list_ax.add_patch(rect)
            file_rects.append(rect)

            # File name text
            filename = os.path.basename(file_path)
            # Truncate long names
            if len(filename) > 28:
                filename = filename[:25] + '...'
            txt = file_list_ax.text(0.03, y_pos, filename,
                                   fontsize=7, verticalalignment='center',
                                   fontweight=fontweight, picker=True, gid=str(i))
            file_text_items.append(txt)

        # Update scrollbar
        if scrollbar_rect[0] is not None:
            scrollbar_rect[0].remove()
        if scrollbar_track_rect[0] is not None:
            scrollbar_track_rect[0].remove()

        # Only show scrollbar if there are more files than can be displayed
        if len(available_folders) > max_visible_files:
            # Calculate scrollbar size and position
            # Height of scrollbar represents visible portion
            scrollbar_height = max_visible_files / len(available_folders)
            # Position represents scroll offset
            scrollbar_position = 1.0 - (file_scroll_offset[0] / len(available_folders)) - scrollbar_height

            from matplotlib.patches import Rectangle
            scrollbar_rect[0] = Rectangle((0.1, scrollbar_position), 0.8, scrollbar_height,
                                         facecolor='gray', edgecolor='darkgray',
                                         linewidth=1, alpha=0.7, picker=True)
            scrollbar_ax.add_patch(scrollbar_rect[0])

            # Add scrollbar track (background) - make it pickable too
            scrollbar_track_rect[0] = Rectangle((0.1, 0), 0.8, 1.0,
                                  facecolor='lightgray', edgecolor='gray',
                                  linewidth=0.5, alpha=0.3, zorder=-1, picker=True)
            scrollbar_ax.add_patch(scrollbar_track_rect[0])

        fig.canvas.draw_idle()

    # Initial display
    update_file_list_display()

    def on_file_list_scroll(event):
        """Handle scroll events on the file list to navigate through files."""
        if event.inaxes != file_list_ax and event.inaxes != scrollbar_ax:
            return

        # Scroll speed: number of files to scroll per wheel notch
        scroll_speed = 3

        # Calculate new scroll offset and update immediately
        if event.button == 'up':
            # Scroll up (show earlier files)
            file_scroll_offset[0] = max(0, file_scroll_offset[0] - scroll_speed)
            update_file_list_display()
        elif event.button == 'down':
            # Scroll down (show later files)
            max_offset = max(0, len(available_folders) - max_visible_files)
            file_scroll_offset[0] = min(max_offset, file_scroll_offset[0] + scroll_speed)
            update_file_list_display()

    def on_scrollbar_press(event):
        """Handle mouse press on scrollbar."""
        if event.inaxes != scrollbar_ax:
            return
        if event.button != 1:  # Only left click
            return

        # Check if we have files that need scrolling
        if len(available_folders) <= max_visible_files:
            return

        is_dragging_scrollbar[0] = True

        # If clicking on track (not on scrollbar itself), jump to that position
        if event.ydata is not None:
            # Convert y position (0-1) to file offset
            # y=1 is top (offset=0), y=0 is bottom (offset=max)
            scrollbar_height = max_visible_files / len(available_folders)
            # Calculate offset from click position
            target_position = 1.0 - event.ydata
            new_offset = target_position * len(available_folders) - (scrollbar_height * len(available_folders) / 2)

            max_offset = len(available_folders) - max_visible_files
            new_offset = int(max(0, min(max_offset, new_offset)))

            file_scroll_offset[0] = new_offset
            update_file_list_display()

    def on_scrollbar_motion(event):
        """Handle mouse motion when dragging scrollbar."""
        if not is_dragging_scrollbar[0]:
            return
        if event.inaxes != scrollbar_ax:
            return
        if len(available_folders) <= max_visible_files:
            return

        # Convert y position to file offset
        if event.ydata is not None:
            # y=1 is top (offset=0), y=0 is bottom (offset=max)
            scrollbar_height = max_visible_files / len(available_folders)
            target_position = 1.0 - event.ydata
            new_offset = target_position * len(available_folders) - (scrollbar_height * len(available_folders) / 2)

            max_offset = len(available_folders) - max_visible_files
            new_offset = int(max(0, min(max_offset, new_offset)))

            file_scroll_offset[0] = new_offset
            update_file_list_display()

    def on_scrollbar_release(event):
        """Handle mouse release after scrollbar drag."""
        if event.button == 1:
            is_dragging_scrollbar[0] = False

    def load_file_by_index(file_idx):
        """Load a folder by its index in the available_folders list."""
        nonlocal measured, T, default_vmin, default_vmax, start_time
        global vmin, vmax

        if file_idx < 0 or file_idx >= len(available_folders):
            return  # Out of bounds

        new_file = available_folders[file_idx]
        if new_file == current_file[0]:
            return  # Same file, do nothing

        # Note: Auto-save disabled when switching files - use 's' key or Save Regions button to save manually
        if combined_mask[0] is not None and combined_mask[0].any():
            print("\nNote: Auto-save disabled. Use 's' key or 'Save Regions' button to save manually.")

        print(f"\nLoading file: {os.path.basename(new_file)}")
        result = load_data_file(new_file)
        if result is None:
            print(f"Failed to load file: {new_file}")
            return

        # Update data
        measured_new, deltas_filename = result
        measured = measured_new
        print(f"Data range: {np.min(measured):.2f} to {np.max(measured):.2f}")
        T = measured.shape[0]

        # Update color scale - fixed range from -800 to 0
        vmin = np.min(measured)
        vmax = np.max(measured)
        default_vmin = vmin
        default_vmax = vmax

        # Detect new frame of interest and get frame differences
        initial_frame, new_frame_diffs, new_diff_threshold = find_last_significant_change(measured, deltas_filename, sensitivity=2.0)

        # Update diff plot with new data
        if diff_plot_elements[0] is not None:  # diff_line exists
            # Remove old plot elements safely
            try:
                diff_plot_elements[0].remove()
            except (ValueError, AttributeError):
                pass
            try:
                diff_plot_elements[1].remove()
            except (ValueError, AttributeError):
                pass
            try:
                diff_plot_elements[2].remove()
            except (ValueError, AttributeError):
                pass

        # Update axes limits and redraw
        diff_plot_ax_ref = diff_plot_elements[3]
        diff_plot_ax_ref.set_xlim(0, T - 1)
        # Set y-axis to show variation better - tight range to emphasize changes
        if len(new_frame_diffs) > 0:
            # Use min and max with some padding to show all data clearly
            y_min = np.min(new_frame_diffs)
            y_max = np.max(new_frame_diffs)
            # Add padding (10% of range) for visual clarity
            y_range = y_max - y_min
            diff_plot_ax_ref.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
        else:
            diff_plot_ax_ref.set_ylim(0, 1)

        # Plot new frame differences
        if len(new_frame_diffs) > 0:
            new_diff_line, = diff_plot_ax_ref.plot(range(len(new_frame_diffs)), new_frame_diffs,
                                                    linewidth=1, color='blue', alpha=0.7)
            new_diff_threshold_line = diff_plot_ax_ref.axhline(y=new_diff_threshold, color='red',
                                                                linestyle='--', linewidth=0.5, alpha=0.7)
            new_diff_current_frame_line = diff_plot_ax_ref.axvline(x=initial_frame, color='green',
                                                                    linewidth=1.5, alpha=0.8)
            # Update references
            diff_plot_elements[0] = new_diff_line
            diff_plot_elements[1] = new_diff_threshold_line
            diff_plot_elements[2] = new_diff_current_frame_line

            # Remove old positive frame markers
            for marker in positive_frame_markers:
                try:
                    marker.remove()
                except (ValueError, AttributeError):
                    pass
            positive_frame_markers.clear()

            # Detect positive frames in the new data
            new_positive_frames = detect_positive_frames(measured)
            print(f"Detected {len(new_positive_frames)} frames with positive values")

            # Add new green markers for frames with positive values
            y_min, y_max = diff_plot_ax_ref.get_ylim()
            marker_height = (y_max - y_min) * 0.15  # 15% of plot height
            marker_y_top = y_max
            marker_y_bottom = y_max - marker_height

            for frame_idx in new_positive_frames:
                if frame_idx < T:  # Ensure frame is within bounds
                    marker_line = diff_plot_ax_ref.plot([frame_idx, frame_idx],
                                                         [marker_y_bottom, marker_y_top],
                                                         color='lime', linewidth=2, alpha=0.6,
                                                         zorder=5)[0]  # zorder=5 to appear above threshold line
                    positive_frame_markers.append(marker_line)

        # Update current file tracking
        current_file[0] = new_file
        old_idx = current_file_idx[0]
        current_file_idx[0] = file_idx

        # Reset timer for new file
        start_time = time.time()
        timer_text.set_text('Time: 0:00')

        # Ensure new file is visible in the list
        if file_idx < file_scroll_offset[0]:
            # File is above visible area, scroll to it
            file_scroll_offset[0] = file_idx
        elif file_idx >= file_scroll_offset[0] + max_visible_files:
            # File is below visible area, scroll to show it at bottom
            new_offset = file_idx - max_visible_files + 1
            file_scroll_offset[0] = new_offset

        # Update file list highlighting
        update_file_list_display()

        # Clear selections
        if region_overlay[0] is not None:
            try:
                if isinstance(region_overlay[0], list):
                    for artist in region_overlay[0]:
                        artist.remove()
                else:
                    region_overlay[0].remove()
            except:
                pass
            region_overlay[0] = None
        combined_mask[0] = None
        region_stack[0].clear()
        regions_saved[0] = False  # Reset save flag for new file

        # Update frame position
        update_frame(initial_frame)

        # Update image
        im.set_array(measured[initial_frame])
        new_norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        im.set_norm(new_norm)
        # Compute max value across all frames for display
        global_max = np.max(measured)
        title.set_text(f"{os.path.basename(new_file)} — frame {initial_frame + 1}/{T} — max: {global_max:.1f}")

        # Remove old cell value texts
        for row in cell_value_texts:
            for txt in row:
                txt.remove()
        cell_value_texts.clear()

        # Recreate cell value texts for new data
        H, W = measured[initial_frame].shape
        for y in range(H):
            row_texts = []
            for x in range(W):
                value = measured[initial_frame][y, x]
                color = 'green' if value > 0 else 'red'
                txt = ax.text(x, y, f'{int(value)}', ha='center', va='center',
                             fontsize=4, color=color, fontweight='bold', visible=show_labels[0])
                row_texts.append(txt)
            cell_value_texts.append(row_texts)

        # Run z-score auto-detection for new file
        print("\nRunning automatic z-score region detection...")
        auto_detect_zscore(None)

        fig.canvas.draw_idle()

    def on_file_click(event):
        """Handle clicks on file list."""
        if event.inaxes != file_list_ax:
            return

        # Find which file was clicked using the gid stored in the rectangle
        for rect in file_rects:
            if rect.contains(event)[0]:
                # Get the actual file index from the gid attribute
                file_idx = int(rect.get_gid())
                load_file_by_index(file_idx)
                break

    def on_diff_plot_click(event):
        """Handle clicks on diff plot to jump to that frame and enable scrubbing."""
        # Check if click is in the diff plot axes
        if event.inaxes != diff_plot_elements[3]:  # diff_plot_ax
            return

        # Only handle left clicks (button 1)
        if event.button != 1:
            return

        # Start scrubbing mode
        is_scrubbing_diff_plot[0] = True

        # Get the x coordinate (frame index) from the click
        if event.xdata is not None:
            frame_idx = int(round(event.xdata))
            # Clamp to valid range
            frame_idx = max(0, min(frame_idx, T - 1))
            # Update frame
            update_frame(frame_idx)
            print(f"Scrubbing: frame {frame_idx + 1}/{T}")

    # Connect file list and diff plot click handlers
    fig.canvas.mpl_connect('button_press_event', on_file_click)
    fig.canvas.mpl_connect('button_press_event', on_diff_plot_click)

    # Timer callback to update the timer display
    def update_timer():
        """Update the timer display."""
        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        timer_text.set_text(f'Time: {minutes}:{seconds:02d}')
        fig.canvas.draw_idle()
        return True  # Keep the timer running

    # Set up timer to update every second
    timer = fig.canvas.new_timer(interval=1000)  # Update every 1000ms (1 second)
    timer.add_callback(update_timer)
    timer.start()

    # Connect event handlers
    fig.canvas.mpl_connect('key_press_event', on_key)
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('button_press_event', on_scrollbar_press)  # Handle scrollbar clicks
    fig.canvas.mpl_connect('motion_notify_event', on_hover)  # Update cursor annotation with blitting
    fig.canvas.mpl_connect('motion_notify_event', on_motion)  # Handle dragging
    fig.canvas.mpl_connect('motion_notify_event', on_scrollbar_motion)  # Handle scrollbar dragging
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('button_release_event', on_scrollbar_release)  # Handle scrollbar release
    fig.canvas.mpl_connect('scroll_event', on_scroll)  # Adjust tolerance with scroll wheel
    fig.canvas.mpl_connect('scroll_event', on_file_list_scroll)  # Scroll through file list
    fig.canvas.mpl_connect('close_event', on_close)

    # Ensure canvas is fully initialized before showing
    # Multiple draw calls and flush to ensure all graphical elements render properly
    print("\nInitializing canvas...")

    # First draw pass - initial setup
    fig.canvas.draw()
    fig.canvas.flush_events()

    # Small delay to allow backend to fully initialize
    import matplotlib
    if matplotlib.get_backend() == 'MacOSX':
        # macOS sometimes needs extra time for renderer initialization
        plt.pause(0.1)

    # Second draw pass - ensure all elements are rendered
    fig.canvas.draw()
    fig.canvas.flush_events()

    # Force tight layout to prevent overlapping elements
    try:
        fig.canvas.draw()
    except:
        pass

    print("Canvas initialization complete.")

    # Run z-score auto-detection by default on startup
    print("\nRunning automatic z-score region detection...")
    auto_detect_zscore(None)

    # Final draw after auto-detection to ensure overlays appear
    fig.canvas.draw()
    fig.canvas.flush_events()

    # show interactive window
    try:
        plt.show()
    except Exception:
        pass

if __name__ == "__main__":
    main()