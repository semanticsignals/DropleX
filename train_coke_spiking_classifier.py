#!/usr/bin/env python3
"""
Train a CNN model for coke spiking classification (per-frame basis).

The model uses:
- CNN layers to extract spatial features from each frame
- Each frame is treated as an independent sample (no temporal modeling)
- 5-fold cross-validation for evaluation
- Region batching: randomly group n regions, average them, then extract each frame as a sample
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from collections import defaultdict
from scipy.ndimage import zoom
from itertools import combinations
import random
import argparse
import os
import sys
import re
import json

# Check for help flags early
if '--help' in sys.argv or '-h' in sys.argv:
    print("""usage: train_coke_spiking_classifier.py [-h] [--batch-size BATCH_SIZE]
                                          [--seed SEED] [--n-regions N_REGIONS]
                                          [--classes CLASSES]
                                          [--binary]

Train coke spiking classifier with CNN (per-frame)

optional arguments:
  -h, --help            show this help message and exit
  --batch-size BATCH_SIZE
                        Batch size for training (default: 16)
  --seed SEED           Random seed for reproducibility (default: 42)
  --n-regions N_REGIONS
                        Number of regions to group and average as one sample (default: 6)
  --classes CLASSES     Comma-separated list of spiking classes to use.
                        Available: unadulterated, alcohol, asa, apap, tylenol
                        (default: all classes)
  --binary              Binary classification: unadulterated vs spiked
                        (default: False - multi-class classification)
  --use-all-combinations
                        Use all C(N,n) combinations (may cause data leakage).
                        Default: False (use non-overlapping random sampling)
  --cross-session       Enable cross-session evaluation mode.
                        Trains on some sessions and tests on completely unseen sessions.
                        (default: False - standard cross-validation)
  --test-sessions SESSIONS
                        Comma-separated list of session names to hold out for testing
                        in cross-session mode. (e.g., "session_coke_unadulterated_2,session_coke_ethanol10_1")
  --day-suffix DAY      Day identifier suffix for data (e.g., '0109' for Jan 9 data).
                        If not specified, uses original data (no date suffix).
                        Use this to train/test exclusively on a specific day's data.
  --sequence-length LENGTH
                        Number of frames to extract from each sample (default: 50).
                        Sequences will be padded or cropped to this length.
                        Can also provide comma-separated list (e.g., "50,20,10,5")
                        to run multiple experiments and generate comparison plot.
  --save-model          Save the best model (PyTorch .pth and ONNX .onnx) for deployment
                        (default: False)
  --model-dir MODEL_DIR
                        Directory to save trained models (default: models/)
  --info                Show detailed information about training
                        configurations""")
    sys.exit(0)

if '--info' in sys.argv:
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║     COKE SPIKING CLASSIFIER TRAINING (PER-FRAME)                     ║
╚══════════════════════════════════════════════════════════════════════╝

This script trains a CNN model to classify coke spiking from
individual sensor frames (no temporal modeling).

──────────────────────────────────────────────────────────────────────
DATA FORMAT
──────────────────────────────────────────────────────────────────────
Expects region files in 'regions/' directory:
- session_coke_unadulterated_regions.npz (or session_coke_unadulterated_*_regions.npz)
- session_coke_alcohol_regions.npz (or session_coke_alcohol_*_regions.npz)
- session_coke_asa_regions.npz (or session_coke_asa_*_regions.npz)
- session_coke_apap_regions.npz (or session_coke_apap_*_regions.npz)
- session_coke_tylenol_regions.npz (or session_coke_tylenol_*_regions.npz)

──────────────────────────────────────────────────────────────────────
CLASSIFICATION MODE
──────────────────────────────────────────────────────────────────────
1. Multi-class classification (default):
   - Classify each spiking type separately
   - Can select specific classes to include

2. Binary classification (--binary):
   - Classify as unadulterated vs spiked
   - All spiking types grouped into one class

──────────────────────────────────────────────────────────────────────
SAMPLING STRATEGY
──────────────────────────────────────────────────────────────────────
Two modes available:

1. Non-overlapping random sampling (default, no data leakage):
   - For N regions, create floor(N/n_regions) non-overlapping groups
   - Each region appears in exactly ONE sample
   - Example: 9 regions, n=2 → 4 samples (groups of 2)
   
2. All combinations (--use-all-combinations, potential data leakage):
   - Generate all C(N, n_regions) combinations of regions
   - Regions can appear in multiple samples (overlapping)
   - Example: 9 regions, n=2 → C(9,2)=36 samples

For each sample, average the n_regions, then extract each frame as a 
separate datapoint for training.

──────────────────────────────────────────────────────────────────────
EXAMPLES
──────────────────────────────────────────────────────────────────────
# Train multi-class with all spiking types (default)
python3 train_coke_spiking_classifier.py

# Train only on specific classes
python3 train_coke_spiking_classifier.py --classes alcohol,asa,apap

# Train with different number of regions
python3 train_coke_spiking_classifier.py --n-regions 4

# Train with all combinations (may have data leakage)
python3 train_coke_spiking_classifier.py --n-regions 2 --use-all-combinations
          
# Train with day suffix
python3 train_coke_spiking_classifier.py --classes unadulterated,ethanol10,ethanol30,ethanol50,ethanol80,ethanol100 --day-suffix 0109 --epochs 20

# Experiment with different sequence lengths
python3 train_coke_spiking_classifier.py --sequence-length 20
python3 train_coke_spiking_classifier.py --sequence-length 10
python3 train_coke_spiking_classifier.py --sequence-length 5

# Run multiple sequence lengths and generate comparison plot
python3 train_coke_spiking_classifier.py --sequence-length 50,20,10,5,2,1
          
# Train and save model for deployment (exports PyTorch .pth and ONNX .onnx)
python3 train_coke_spiking_classifier.py --classes unadulterated,ethanol10,ethanol30,ethanol50,ethanol80,ethanol100 --epochs 10 --day-suffix 0109 --sequence-length 50 --save-model

# Save model to custom directory
python3 train_coke_spiking_classifier.py --classes unadulterated,ethanol10,ethanol30,ethanol50,ethanol80,ethanol100 --epochs 10 --day-suffix 0109 --sequence-length 50 --save-model --model-dir my_models
══════════════════════════════════════════════════════════════════════
""")
    sys.exit(0)

REGIONS_DIR = 'regions'

# All available spiking types
ALL_SPIKINGS = ['unadulterated', 'alcohol', 'asa', 'apap', 'tylenol', 'ethanol5', 'ethanol20', 'ethanol10', 'ethanol30', 'ethanol40', 'ethanol50', 'ethanol60', 'ethanol80', 'ethanol100']

def get_binary_label(spiking):
    """Convert spiking type to binary label for unadulterated vs spiked classification."""
    return 0 if spiking == 'unadulterated' else 1

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_region_data(npz_file):
    """
    Load region data and compute average centroid value for each region (mean over frames).
    Returns list of region averages and their corresponding patches.
    """
    data = np.load(npz_file)
    num_regions = int(data['num_regions'])
    region_patches = []
    
    for region_idx in range(num_regions):
        crops_key = f'region_{region_idx}_crops'
        mask_key = f'region_{region_idx}_crop_mask'
        if crops_key in data and mask_key in data:
            crops = data[crops_key]  # (num_frames, H, W)
            crop_mask = data[mask_key]  # (H, W) boolean mask
            
            # Store the entire temporal sequence for this region
            region_patches.append(crops)
    
    return region_patches

class SpikingDataset(Dataset):
    """Dataset for coke spiking classification from temporal patches."""

    def __init__(self, patches, labels, spikings, normalize=True, mean=None, std=None):
        self.patches = torch.FloatTensor(patches)
        self.labels = torch.LongTensor(labels)
        self.spikings = spikings
        
        if normalize:
            if mean is None:
                self.mean = self.patches.mean()
                self.std = self.patches.std()
            else:
                self.mean = mean
                self.std = std
            self.patches = (self.patches - self.mean) / (self.std + 1e-8)
        else:
            self.mean = 0
            self.std = 1

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        return self.patches[idx], self.labels[idx]

class SpatioTemporalCNN(nn.Module):
    """Simple CNN model for single-frame classification (no temporal modeling)."""

    def __init__(self, input_size=7, num_classes=4, hidden_dim=64, lstm_layers=2):
        super(SpatioTemporalCNN, self).__init__()
        
        # CNN layers for spatial feature extraction
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        
        # Calculate CNN output size after two pooling operations
        conv_output_size = 64 * ((input_size // 4) ** 2)
        
        # Fully connected layers for classification
        self.fc1 = nn.Linear(conv_output_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x shape: (batch_size, H, W) - single frames
        batch_size, H, W = x.size()
        
        # Add channel dimension
        x = x.view(batch_size, 1, H, W)
        
        # Process through CNN
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        # Flatten
        x = x.view(batch_size, -1)
        
        # Classification
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        out = self.fc2(x)
        return out

class FrameAveragingCNN(nn.Module):
    """Simple CNN model for single-frame classification (alternative architecture)."""

    def __init__(self, input_size=7, num_classes=4):
        super(FrameAveragingCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        
        conv_output_size = 64 * ((input_size // 4) ** 2)
        self.fc1 = nn.Linear(conv_output_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x shape: (batch_size, H, W) - single frames
        batch_size, H, W = x.size()
        
        # Add channel dimension
        x = x.view(batch_size, 1, H, W)
        
        # Process through CNN
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        # Flatten
        x = x.view(batch_size, -1)
        
        # Classification
        out = torch.relu(self.fc1(x))
        out = self.dropout(out)
        out = self.fc2(out)
        return out

def load_coke_spiking_data(patches_dir="regions", sequence_length=50, n_regions=6, 
                           use_all_combinations=False, selected_classes=None, binary=False, day_suffix=None):
    """
    Load coke spiking patch data and create samples by batching regions.
    
    Parameters:
    -----------
    patches_dir : str
        Directory containing region files
    sequence_length : int
        Fixed sequence length (will crop or pad to this length)
    n_regions : int
        Number of regions to randomly group and average as one sample
    use_all_combinations : bool
        If True, use all C(N, n_regions) combinations (may cause data leakage).
        If False, use non-overlapping random sampling (no data leakage).
    selected_classes : list
        List of spiking types to include. If None, use all available.
    binary : bool
        If True, perform binary classification (unadulterated vs spiked).
        If False, perform multi-class classification.
    day_suffix : str or None
        Day identifier suffix (e.g., '0109' for Jan 9 data). If None, uses original data (no date suffix).
    
    Returns:
    --------
    data : dict
        Contains 'combinations', 'combo_labels', 'combo_spikings', 
        'spiking_names', 'label_to_spiking', 'sequence_length',
        'combo_sessions' (session identifier for each combination)
    """
    patches_dir = Path(patches_dir)
    
    # Determine which classes to load
    if binary:
        # For binary classification, we need unadulterated and selected spiking types (or all if none selected)
        if selected_classes is None:
            spikings_to_load = ALL_SPIKINGS
            print("Binary classification mode: unadulterated vs spiked (all spiking types)")
        else:
            # Ensure unadulterated is included for binary classification
            spikings_to_load = selected_classes if 'unadulterated' in selected_classes else ['unadulterated'] + selected_classes
            print(f"Binary classification mode: unadulterated vs spiked")
            print(f"Loading spiking types: {spikings_to_load}")
    elif selected_classes is None:
        spikings_to_load = ALL_SPIKINGS
        print("Multi-class classification mode")
    else:
        spikings_to_load = selected_classes
        print(f"Loading spiking types: {spikings_to_load}")
        print("Multi-class classification mode")
    
    # Group by spiking type
    spiking_data = defaultdict(list)
    original_lengths = []
    # Track which session each region comes from
    spiking_data_sessions = defaultdict(list)
    # Track file creation dates
    file_dates = {}  # filename -> creation date
    
    for spiking in spikings_to_load:
        # Find all files matching this spiking type (handles multiple sessions)
        # Match both patterns: with and without session number
        # e.g., session_coke_alcohol_regions.npz AND session_coke_alcohol_2_regions.npz
        
        # Special case: ethanol100 data is stored as conc_alcohol_100
        if spiking == 'ethanol100':
            if day_suffix:
                pattern1 = f'session_conc_alcohol_100_{day_suffix}_regions.npz'
                pattern2 = f'session_conc_alcohol_100_{day_suffix}_*_regions.npz'
            else:
                pattern1 = f'session_conc_alcohol_100_regions.npz'  # without number
                pattern2 = f'session_conc_alcohol_100_*_regions.npz'  # with number
        else:
            if day_suffix:
                pattern1 = f'session_coke_{spiking}_{day_suffix}_regions.npz'
                pattern2 = f'session_coke_{spiking}_{day_suffix}_*_regions.npz'
            else:
                pattern1 = f'session_coke_{spiking}_regions.npz'  # without number
                pattern2 = f'session_coke_{spiking}_*_regions.npz'  # with number
        
        matching_files = []
        # Check for file without number
        file_without_number = patches_dir / pattern1
        if file_without_number.exists():
            matching_files.append(file_without_number)
        
        # Check for files with numbers
        matching_files.extend([f for f in patches_dir.glob(pattern2) if f != file_without_number])
        
        # Filter out files that contain "frame" in the name (skip frame-specific data)
        matching_files = [f for f in matching_files if 'frame' not in f.name.lower()]
        
        # Filter based on day_suffix parameter
        if day_suffix:
            # Keep only files with the specified day suffix
            filtered_files = []
            for f in matching_files:
                name_parts = f.stem.split('_')
                # Check if the day_suffix is in the filename
                if day_suffix in name_parts:
                    filtered_files.append(f)
        else:
            # Filter out files with date suffixes (e.g., '0109') by default
            # Date suffixes are 4-digit patterns that look like dates (MMDD format)
            filtered_files = []
            for f in matching_files:
                name_parts = f.stem.split('_')
                # Check if any part is a 4-digit number (likely a date suffix like '0109')
                has_date_suffix = any(len(part) == 4 and part.isdigit() for part in name_parts)
                if not has_date_suffix:
                    filtered_files.append(f)
        
        matching_files = sorted(filtered_files)
        
        if not matching_files:
            print(f"Warning: No files found for {spiking}, skipping")
            continue
        
        if day_suffix:
            print(f"  Found {len(matching_files)} file(s) for {spiking} (day: {day_suffix})")
        else:
            print(f"  Found {len(matching_files)} file(s) for {spiking} (excluding frame-specific and date-suffixed files)")
        
        # Load data from all matching files
        for file in matching_files:
            # Get file creation date
            import os
            from datetime import datetime
            creation_time = os.path.getctime(file)
            creation_date = datetime.fromtimestamp(creation_time).strftime('%Y-%m-%d')
            file_dates[file.name] = creation_date
            
            region_patches = load_region_data(file)
            session_name = file.stem  # e.g., "session_coke_ethanol10_2"
            for patch in region_patches:
                original_lengths.append(len(patch))
                # Normalize sequence length
                if len(patch) < sequence_length:
                    # Pad with last frame
                    pad_length = sequence_length - len(patch)
                    patch = np.concatenate([patch, np.repeat(patch[-1:], pad_length, axis=0)], axis=0)
                else:
                    # Crop to sequence length
                    patch = patch[:sequence_length]
                spiking_data[spiking].append(patch)
                spiking_data_sessions[spiking].append(session_name)
    
    if not spiking_data:
        raise ValueError("No data loaded! Check that the region files exist.")
    
    print(f"Loaded data for: {sorted(spiking_data.keys())}")
    print(f"Original sequence lengths: min={min(original_lengths)}, max={max(original_lengths)}, mean={np.mean(original_lengths):.1f}")
    
    # Print data collection timeline
    print(f"\n{'='*60}")
    print("DATA COLLECTION TIMELINE")
    print(f"{'='*60}")
    
    # Group files by date
    from collections import defaultdict as dd
    files_by_date = dd(list)
    for filename, date in sorted(file_dates.items()):
        files_by_date[date].append(filename)
    
    # Print summary by date
    day_counter = 1
    for date in sorted(files_by_date.keys()):
        files = files_by_date[date]
        print(f"\nDay {day_counter} ({date}): {len(files)} session(s)")
        
        # Count regions per spiking type for this day
        day_spiking_counts = defaultdict(int)
        for filename in files:
            # Extract spiking type from filename
            for spiking in spikings_to_load:
                if f'_{spiking}_' in filename or filename.startswith(f'session_coke_{spiking}_'):
                    # Count how many regions this file contributed
                    session_name = filename.replace('.npz', '')
                    # Count occurrences in spiking_data_sessions
                    for sp, sessions in spiking_data_sessions.items():
                        count = sessions.count(session_name)
                        if count > 0 and sp == spiking:
                            day_spiking_counts[sp] += count
                    break
        
        # Print breakdown by spiking type
        for spiking in sorted(day_spiking_counts.keys()):
            print(f"  {spiking}: {day_spiking_counts[spiking]} regions")
        
        day_counter += 1
    
    print(f"\nTotal: {len(file_dates)} files across {len(files_by_date)} day(s)")
    print(f"{'='*60}\n")
    
    # Determine common spatial dimensions (use the maximum dimensions found)
    all_spatial_dims = []
    for spiking, patches in spiking_data.items():
        for patch in patches:
            all_spatial_dims.append(patch.shape[1:])  # (H, W)
    
    max_h = max(dim[0] for dim in all_spatial_dims)
    max_w = max(dim[1] for dim in all_spatial_dims)
    print(f"Standardizing spatial dimensions to: ({max_h}, {max_w})")
    
    # Resize all patches to common dimensions
    for spiking in spiking_data.keys():
        resized_patches = []
        for patch in spiking_data[spiking]:
            if patch.shape[1] != max_h or patch.shape[2] != max_w:
                # Resize spatial dimensions
                zoom_factors = (1.0, max_h / patch.shape[1], max_w / patch.shape[2])
                resized_patch = zoom(patch, zoom_factors, order=1)
                resized_patches.append(resized_patch)
            else:
                resized_patches.append(patch)
        spiking_data[spiking] = resized_patches
    
    # Create labels
    if binary:
        # Binary classification: unadulterated (0) vs spiked (1)
        spiking_names = ['unadulterated', 'spiked']
        label_to_spiking = {0: 'unadulterated', 1: 'spiked'}
        spiking_to_label = {s: get_binary_label(s) for s in spiking_data.keys()}
    else:
        # Multi-class classification
        # Preserve order from spikings_to_load (which comes from --classes argument or ALL_SPIKINGS)
        # Filter to only include classes that actually have data loaded
        spiking_names = [s for s in spikings_to_load if s in spiking_data]
        label_to_spiking = {i: spiking for i, spiking in enumerate(spiking_names)}
        spiking_to_label = {spiking: i for i, spiking in enumerate(spiking_names)}
    
    # Now create samples by generating combinations of n_regions
    # Store combinations first, then split, then extract frames to avoid data leakage
    all_combinations = []
    all_combo_labels = []
    all_combo_spikings = []
    all_combo_sessions = []  # Track which session each combination comes from
    
    for spiking, patches in spiking_data.items():
        print(f"\n{spiking}: {len(patches)} regions")
        
        # Check if we have enough regions
        if len(patches) < n_regions:
            print(f"  Warning: Only {len(patches)} regions available, need at least {n_regions}")
            print(f"  Skipping this spiking type")
            continue
        
        label = spiking_to_label[spiking]
        sessions = spiking_data_sessions[spiking]
        
        if use_all_combinations:
            # Generate all combinations of n_regions from available patches
            all_combos = list(combinations(range(len(patches)), n_regions))
            print(f"  Using all combinations: C({len(patches)}, {n_regions}) = {len(all_combos)}")
            print(f"  WARNING: This may cause data leakage due to overlapping regions!")
            
            # Create averaged sequence for each combination
            for combo_indices in all_combos:
                # Get the patches for this combination
                combo_patches = [patches[i] for i in combo_indices]
                combo_session_names = [sessions[i] for i in combo_indices]
                
                # Average the patches
                combo_array = np.array(combo_patches)  # (n_regions, seq_len, H, W)
                averaged_sample = np.mean(combo_array, axis=0)  # (seq_len, H, W)
                
                # Store the combination
                all_combinations.append(averaged_sample)
                all_combo_labels.append(label)
                all_combo_spikings.append(spiking)
                # For session tracking, use the first session in the combination
                all_combo_sessions.append(combo_session_names[0])
            
            print(f"  Created {len(all_combos)} combinations (will expand to {len(all_combos) * sequence_length} frames)")
        
        else:
            # Non-overlapping random sampling (no data leakage)
            num_samples = len(patches) // n_regions
            print(f"  Using non-overlapping sampling: {len(patches)} // {n_regions} = {num_samples} samples")
            print(f"  No data leakage: each region appears in exactly ONE sample")
            
            # Shuffle region indices
            indices = list(range(len(patches)))
            random.shuffle(indices)
            
            # Create non-overlapping groups
            for i in range(num_samples):
                start_idx = i * n_regions
                end_idx = start_idx + n_regions
                combo_indices = indices[start_idx:end_idx]
                
                # Get the patches for this combination
                combo_patches = [patches[idx] for idx in combo_indices]
                combo_session_names = [sessions[idx] for idx in combo_indices]
                
                # Average the patches
                combo_array = np.array(combo_patches)  # (n_regions, seq_len, H, W)
                averaged_sample = np.mean(combo_array, axis=0)  # (seq_len, H, W)
                
                # Store the combination
                all_combinations.append(averaged_sample)
                all_combo_labels.append(label)
                all_combo_spikings.append(spiking)
                # For session tracking, use the first session in the combination
                all_combo_sessions.append(combo_session_names[0])
            
            print(f"  Created {num_samples} non-overlapping samples (will expand to {num_samples * sequence_length} frames)")
    
    # Convert combinations to numpy arrays
    all_combinations = np.array(all_combinations)  # (num_combos, seq_len, H, W)
    all_combo_labels = np.array(all_combo_labels)
    all_combo_spikings = np.array(all_combo_spikings)
    all_combo_sessions = np.array(all_combo_sessions)
    
    print(f"\nCombination summary:")
    print(f"  Total combinations: {len(all_combinations)}")
    print(f"  Combination shape: {all_combinations.shape}")
    print(f"  Number of classes: {len(spiking_names)}")
    print(f"  Combinations per class:")
    for i, name in enumerate(spiking_names):
        count = np.sum(all_combo_labels == i)
        print(f"    {name}: {count} combinations")
    
    # Print unique sessions
    unique_sessions = np.unique(all_combo_sessions)
    print(f"\n  Unique sessions: {len(unique_sessions)}")
    for session in sorted(unique_sessions):
        count = np.sum(all_combo_sessions == session)
        print(f"    {session}: {count} combinations")
    
    return {
        'combinations': all_combinations,  # (num_combos, seq_len, H, W)
        'combo_labels': all_combo_labels,
        'combo_spikings': all_combo_spikings,
        'combo_sessions': all_combo_sessions,
        'spiking_names': spiking_names,
        'label_to_spiking': label_to_spiking,
        'sequence_length': sequence_length
    }

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for patches, labels in train_loader:
        patches, labels = patches.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(patches)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(train_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy

def evaluate(model, test_loader, criterion, device, return_probabilities=False):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for patches, labels in test_loader:
            patches, labels = patches.to(device), labels.to(device)
            outputs = model(patches)
            loss = criterion(outputs, labels)
            
            # Get probabilities using softmax
            probs = torch.softmax(outputs, dim=1)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    avg_loss = total_loss / len(test_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    if return_probabilities:
        return avg_loss, accuracy, all_preds, all_labels, np.array(all_probs)
    return avg_loss, accuracy, all_preds, all_labels

def train_cross_validation(data, n_folds=5, epochs=100, batch_size=16, lr=0.001, save_model=False, model_save_path=None):
    """Train and evaluate model with k-fold cross-validation.
    
    Args:
        data: Dictionary containing combinations, labels, etc.
        n_folds: Number of folds for cross-validation
        epochs: Number of training epochs
        batch_size: Training batch size
        lr: Learning rate
        save_model: Whether to save the best model across all folds
        model_save_path: Path to save the model (without extension)
    """
    combinations = data['combinations']  # (num_combos, seq_len, H, W)
    combo_labels = data['combo_labels']
    combo_spikings = data['combo_spikings']  # Original spiking types before binary conversion
    spiking_names = data['spiking_names']
    sequence_length = data['sequence_length']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set up k-fold cross-validation at the COMBINATION level to avoid data leakage
    print(f"\n=== {n_folds}-Fold Cross-Validation (no data leakage) ===")
    
    num_classes = len(spiking_names)
    
    # Check if we can use stratified k-fold
    # Need at least n_folds samples per class
    class_counts = [np.sum(combo_labels == i) for i in range(num_classes)]
    min_class_count = min(class_counts)
    
    if min_class_count >= n_folds:
        print(f"Using Stratified {n_folds}-Fold CV")
        kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        splits = list(kfold.split(combinations, combo_labels))
    else:
        print(f"Warning: Smallest class has only {min_class_count} samples, need at least {n_folds}")
        print(f"Using regular {n_folds}-Fold CV instead")
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        splits = list(kfold.split(combinations))
    
    # Store results for each fold
    fold_results = []
    all_test_preds = []
    all_test_true = []
    all_test_probs = []
    all_test_spikings = []  # Store original spiking types
    
    # Track best model across all folds
    best_model_state = None
    best_model_acc = 0
    best_model_fold = 0
    best_model_input_size = None
    best_model_normalization = None  # Store mean and std for normalization

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx + 1}/{n_folds}")
        print(f"{'='*60}")
        
        # Split combinations
        combo_train = combinations[train_idx]
        combo_test = combinations[test_idx]
        label_train = combo_labels[train_idx]
        label_test = combo_labels[test_idx]
        spiking_train = combo_spikings[train_idx]
        spiking_test = combo_spikings[test_idx]

        print(f"Train combinations: {len(combo_train)}")
        print(f"Test combinations: {len(combo_test)}")

        # Extract frames from each split
        X_train = []
        y_train = []
        for combo, label in zip(combo_train, label_train):
            for frame_idx in range(combo.shape[0]):
                X_train.append(combo[frame_idx])
                y_train.append(label)

        X_test = []
        y_test = []
        y_test_spikings = []
        for combo, label, spiking in zip(combo_test, label_test, spiking_test):
            for frame_idx in range(combo.shape[0]):
                X_test.append(combo[frame_idx])
                y_test.append(label)
                y_test_spikings.append(spiking)
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        
        print(f"Train frames: {len(X_train)} (from {len(combo_train)} combos × {sequence_length} frames)")
        print(f"Test frames: {len(X_test)} (from {len(combo_test)} combos × {sequence_length} frames)")
        
        # Create datasets
        train_dataset = SpikingDataset(X_train, y_train, spiking_names)
        test_dataset = SpikingDataset(X_test, y_test, spiking_names,
                                      mean=train_dataset.mean, std=train_dataset.std)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model for this fold
        input_size = X_train.shape[1]
        model = SpatioTemporalCNN(input_size=input_size, num_classes=num_classes)
        model = model.to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Learning rate scheduler - reduces LR when validation loss plateaus
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=False, min_lr=1e-6
        )
        
        # Training loop
        train_losses = []
        test_losses = []
        train_accs = []
        test_accs = []
        best_test_acc = 0
        best_epoch = 0
        best_fold_model_state = None  # Track best model state within this fold
        
        print(f"\nTraining fold {fold_idx + 1}...")
        
        for epoch in range(epochs):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            test_loss, test_acc, _, _ = evaluate(model, test_loader, criterion, device)
            
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            
            # Step the scheduler based on validation loss
            scheduler.step(test_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_epoch = epoch + 1
                # Save best model state for this fold
                best_fold_model_state = {
                    'model_state_dict': model.state_dict(),
                    'input_size': input_size,
                    'num_classes': num_classes,
                    'epoch': epoch + 1,
                    'accuracy': test_acc
                }
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch [{epoch+1}/{epochs}] "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                      f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f} | "
                      f"LR: {current_lr:.2e}")
        
        # Check if this fold has the best model across all folds
        if best_test_acc > best_model_acc:
            best_model_acc = best_test_acc
            best_model_fold = fold_idx + 1
            best_model_state = best_fold_model_state
            best_model_input_size = input_size
            best_model_normalization = {
                'mean': train_dataset.mean,
                'std': train_dataset.std
            }
        
        # Final evaluation with probabilities
        _, final_test_acc, test_preds, test_true, test_probs = evaluate(
            model, test_loader, criterion, device, return_probabilities=True
        )
        
        print(f"\nFold {fold_idx + 1} Results:")
        print(f"  Best test accuracy: {best_test_acc:.4f} at epoch {best_epoch}")
        print(f"  Final test accuracy: {final_test_acc:.4f}")
        
        # Store fold results
        fold_results.append({
            'fold': fold_idx + 1,
            'train_losses': train_losses,
            'test_losses': test_losses,
            'train_accs': train_accs,
            'test_accs': test_accs,
            'best_test_acc': best_test_acc,
            'final_test_acc': final_test_acc,
            'test_preds': test_preds,
            'test_true': test_true,
            'test_probs': test_probs
        })
        
        # Accumulate all predictions for overall metrics
        all_test_preds.extend(test_preds)
        all_test_true.extend(test_true)
        all_test_probs.extend(test_probs)
        all_test_spikings.extend(y_test_spikings)
    
    # Compute overall statistics
    all_test_preds = np.array(all_test_preds)
    all_test_true = np.array(all_test_true)
    all_test_probs = np.array(all_test_probs)
    all_test_spikings = np.array(all_test_spikings)
    
    # Calculate average metrics across folds
    avg_best_acc = np.mean([r['best_test_acc'] for r in fold_results])
    std_best_acc = np.std([r['best_test_acc'] for r in fold_results])
    avg_final_acc = np.mean([r['final_test_acc'] for r in fold_results])
    std_final_acc = np.std([r['final_test_acc'] for r in fold_results])
    
    print(f"\n{'='*60}")
    print(f"CROSS-VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"Average best test accuracy: {avg_best_acc:.4f} ± {std_best_acc:.4f}")
    print(f"Average final test accuracy: {avg_final_acc:.4f} ± {std_final_acc:.4f}")
    
    # Print per-fold results
    print(f"\nPer-fold results:")
    for result in fold_results:
        print(f"  Fold {result['fold']}: Best={result['best_test_acc']:.4f}, Final={result['final_test_acc']:.4f}")
    
    # Overall accuracy (all predictions combined)
    overall_acc = accuracy_score(all_test_true, all_test_preds)
    print(f"\nOverall accuracy (all folds combined): {overall_acc:.4f}")
    
    # Save the best model if requested
    if save_model and best_model_state is not None and model_save_path is not None:
        print(f"\n{'='*60}")
        print(f"SAVING BEST MODEL")
        print(f"{'='*60}")
        print(f"Best model from Fold {best_model_fold} with accuracy: {best_model_acc:.4f}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_save_path) if os.path.dirname(model_save_path) else '.', exist_ok=True)
        
        # Save PyTorch model
        torch_path = f"{model_save_path}.pth"
        torch.save({
            'model_state_dict': best_model_state['model_state_dict'],
            'input_size': best_model_state['input_size'],
            'num_classes': best_model_state['num_classes'],
            'spiking_names': spiking_names,
            'sequence_length': sequence_length,
            'best_accuracy': best_model_acc,
            'best_fold': best_model_fold,
            'best_epoch': best_model_state['epoch'],
            'normalization_mean': float(best_model_normalization['mean']),
            'normalization_std': float(best_model_normalization['std'])
        }, torch_path)
        print(f"PyTorch model saved to: {torch_path}")
        
        # Save metadata as JSON for easy deployment (do this before ONNX export)
        metadata_path = f"{model_save_path}_metadata.json"
        metadata = {
            'model_type': 'SpatioTemporalCNN',
            'input_size': best_model_state['input_size'],
            'num_classes': best_model_state['num_classes'],
            'spiking_names': spiking_names,
            'sequence_length': sequence_length,
            'best_accuracy': float(best_model_acc),
            'best_fold': int(best_model_fold),
            'best_epoch': int(best_model_state['epoch']),
            'normalization': {
                'mean': float(best_model_normalization['mean']),
                'std': float(best_model_normalization['std'])
            },
            'input_shape': [best_model_state['input_size'], best_model_state['input_size']],
            'output_shape': [best_model_state['num_classes']],
            'description': 'Coke spiking classifier - per-frame CNN model'
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Model metadata saved to: {metadata_path}")
        
        # Export to ONNX
        print("\nAttempting ONNX export...")
        try:
            # Load the best model for ONNX export
            export_model = SpatioTemporalCNN(
                input_size=best_model_state['input_size'],
                num_classes=best_model_state['num_classes']
            )
            export_model.load_state_dict(best_model_state['model_state_dict'])
            export_model.eval()
            
            # Move to CPU for ONNX export (more compatible)
            export_model = export_model.cpu()
            
            # Create a dummy input for tracing
            # Shape: (batch_size=1, H, W)
            dummy_input = torch.randn(1, best_model_state['input_size'], best_model_state['input_size'])
            
            print(f"  Model input size: {best_model_state['input_size']}x{best_model_state['input_size']}")
            print(f"  Number of classes: {best_model_state['num_classes']}")
            print(f"  Dummy input shape: {dummy_input.shape}")
            
            # Test forward pass before export
            with torch.no_grad():
                test_output = export_model(dummy_input)
                print(f"  Test output shape: {test_output.shape}")
            
            # Export to ONNX
            onnx_path = f"{model_save_path}.onnx"
            print(f"  Exporting to: {onnx_path}")
            
            torch.onnx.export(
                export_model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                },
                verbose=False
            )
            print(f"✓ ONNX model saved to: {onnx_path}")
            
            # Verify ONNX model
            try:
                import onnx
                onnx_model = onnx.load(onnx_path)
                onnx.checker.check_model(onnx_model)
                print(f"✓ ONNX model verification passed")
            except ImportError:
                print("  (Note: Install 'onnx' package to verify exported model)")
            except Exception as verify_error:
                print(f"  Warning: ONNX model verification failed: {verify_error}")
            
            print(f"\n{'='*60}")
            print(f"MODEL EXPORT COMPLETE!")
            print(f"{'='*60}")
            print(f"  PyTorch model:  {torch_path}")
            print(f"  ONNX model:     {onnx_path}")
            print(f"  Metadata JSON:  {metadata_path}")
            print(f"{'='*60}")
            
        except Exception as e:
            print(f"\n{'='*60}")
            print(f"WARNING: ONNX export failed")
            print(f"{'='*60}")
            print(f"Error: {e}")
            print(f"\nThe PyTorch model was saved successfully at: {torch_path}")
            print(f"You can still use the PyTorch model for inference.")
            print(f"\nTo fix ONNX export issues, ensure you have:")
            print(f"  - torch >= 1.9.0")
            print(f"  - onnx (optional, for verification)")
            print(f"{'='*60}")
            import traceback
            traceback.print_exc()
    
    # Print average confidence scores per class
    print(f"\n{'='*60}")
    print(f"AVERAGE CONFIDENCE SCORES PER CLASS (ALL FOLDS)")
    print(f"{'='*60}")
    for class_idx, spiking_name in enumerate(spiking_names):
        class_mask = (all_test_true == class_idx)
        num_samples = np.sum(class_mask)
        print(f"\nClass {class_idx} ({spiking_name}): {num_samples} samples")
        if num_samples > 0:
            class_probs = all_test_probs[class_mask]
            avg_confidence = np.mean(class_probs[:, class_idx])
            print(f"  Average confidence: {avg_confidence:.4f}")
    
    return {
        'fold_results': fold_results,
        'avg_best_acc': avg_best_acc,
        'std_best_acc': std_best_acc,
        'avg_final_acc': avg_final_acc,
        'std_final_acc': std_final_acc,
        'overall_acc': overall_acc,
        'all_test_preds': all_test_preds,
        'all_test_true': all_test_true,
        'all_test_probs': all_test_probs,
        'all_test_spikings': all_test_spikings,
        'spiking_names': spiking_names,
        'n_folds': n_folds,
        'sequence_length': sequence_length
    }


def train_cross_session(data, test_sessions=None, epochs=100, batch_size=16, lr=0.001):
    """Train and evaluate model using cross-session validation.
    
    Args:
        data: Dictionary containing combinations, labels, sessions, etc.
        test_sessions: List of session names to use for testing. If None, automatically select 20% of sessions.
        epochs: Number of training epochs
        batch_size: Training batch size
        lr: Learning rate
    
    Returns:
        Dictionary with training results, predictions, and metrics
    """
    combinations = data['combinations']  # (num_combos, seq_len, H, W)
    combo_labels = data['combo_labels']
    combo_spikings = data['combo_spikings']  # Original spiking types before binary conversion
    combo_sessions = data['combo_sessions']
    spiking_names = data['spiking_names']
    sequence_length = data['sequence_length']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print(f"\n=== Cross-Session Validation ===")
    
    num_classes = len(spiking_names)
    unique_sessions = np.unique(combo_sessions)
    
    # Group sessions by spiking type to ensure balanced splits
    sessions_by_spiking = defaultdict(list)
    for session in unique_sessions:
        session_mask = (combo_sessions == session)
        # Determine the spiking type for this session (use the most common one)
        session_spikings = combo_spikings[session_mask]
        most_common_spiking = max(set(session_spikings), key=lambda x: list(session_spikings).count(x))
        sessions_by_spiking[most_common_spiking].append(session)
    
    print(f"\nSessions grouped by spiking type:")
    for spiking, sessions in sorted(sessions_by_spiking.items()):
        print(f"  {spiking}: {len(sessions)} session(s)")
    
    # Determine test sessions
    if test_sessions is None:
        # Automatically select sessions ensuring at least 1 per class in both train and test
        test_sessions = []
        train_sessions = []
        
        np.random.seed(42)
        
        # For each spiking type, select at least 1 for test and keep at least 1 for train
        for spiking, sessions in sessions_by_spiking.items():
            if len(sessions) < 2:
                print(f"Warning: Only {len(sessions)} session(s) for {spiking}, cannot split properly")
                print(f"         Putting this session in train set only")
                train_sessions.extend(sessions)
            else:
                # Shuffle sessions for this spiking type
                sessions_shuffled = list(sessions)
                np.random.shuffle(sessions_shuffled)
                
                # Put at least 1 in test, rest in train
                num_test = max(1, int(len(sessions) * 0.2))
                test_sessions.extend(sessions_shuffled[:num_test])
                train_sessions.extend(sessions_shuffled[num_test:])
        
        print(f"\nAutomatically selected {len(test_sessions)} test sessions (stratified by class):")
        for spiking in sorted(sessions_by_spiking.keys()):
            test_for_spiking = [s for s in test_sessions if s in sessions_by_spiking[spiking]]
            train_for_spiking = [s for s in train_sessions if s in sessions_by_spiking[spiking]]
            print(f"  {spiking}: {len(test_for_spiking)} test, {len(train_for_spiking)} train")
    else:
        # Validate provided test sessions
        invalid_sessions = [s for s in test_sessions if s not in unique_sessions]
        if invalid_sessions:
            print(f"Error: Invalid test sessions: {invalid_sessions}")
            print(f"Available sessions: {sorted(unique_sessions)}")
            return None
        
        # Check if each class has at least one session in test and train
        train_sessions = [s for s in unique_sessions if s not in test_sessions]
        
        # Verify coverage
        test_spikings = set()
        train_spikings = set()
        
        for spiking, sessions in sessions_by_spiking.items():
            if any(s in test_sessions for s in sessions):
                test_spikings.add(spiking)
            if any(s in train_sessions for s in sessions):
                train_spikings.add(spiking)
        
        missing_in_test = set(sessions_by_spiking.keys()) - test_spikings
        missing_in_train = set(sessions_by_spiking.keys()) - train_spikings
        
        if missing_in_test:
            print(f"Warning: Classes missing in test set: {missing_in_test}")
        if missing_in_train:
            print(f"Error: Classes missing in train set: {missing_in_train}")
            print(f"Cannot train without samples from all classes!")
            return None
        
        print(f"Using provided test sessions: {sorted(test_sessions)}")
    
    # Split combinations by session
    test_mask = np.isin(combo_sessions, test_sessions)
    train_mask = ~test_mask
    
    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]
    
    combo_train = combinations[train_idx]
    combo_test = combinations[test_idx]
    label_train = combo_labels[train_idx]
    label_test = combo_labels[test_idx]
    spiking_train = combo_spikings[train_idx]
    spiking_test = combo_spikings[test_idx]
    session_train = combo_sessions[train_idx]
    session_test = combo_sessions[test_idx]
    
    # Print split information
    actual_train_sessions = np.unique(session_train)
    actual_test_sessions = np.unique(session_test)
    
    print(f"\nTrain sessions ({len(actual_train_sessions)}): {sorted(actual_train_sessions)}")
    print(f"  Train combinations: {len(combo_train)}")
    for i, name in enumerate(spiking_names):
        count = np.sum(label_train == i)
        print(f"    {name}: {count} combinations")
    
    print(f"\nTest sessions ({len(actual_test_sessions)}): {sorted(actual_test_sessions)}")
    print(f"  Test combinations: {len(combo_test)}")
    for i, name in enumerate(spiking_names):
        count = np.sum(label_test == i)
        print(f"    {name}: {count} combinations")
    
    # Verify that all classes have at least one sample in both train and test
    train_classes_present = set()
    test_classes_present = set()
    for i, name in enumerate(spiking_names):
        if np.sum(label_train == i) > 0:
            train_classes_present.add(name)
        if np.sum(label_test == i) > 0:
            test_classes_present.add(name)
    
    missing_in_train = set(spiking_names) - train_classes_present
    missing_in_test = set(spiking_names) - test_classes_present
    
    if missing_in_train:
        print(f"\nError: Classes missing in train set: {missing_in_train}")
        return None
    if missing_in_test:
        print(f"\nWarning: Classes missing in test set: {missing_in_test}")
        print(f"This may affect evaluation metrics for these classes.")
    
    # Extract frames from each split
    X_train = []
    y_train = []
    for combo, label in zip(combo_train, label_train):
        for frame_idx in range(combo.shape[0]):
            X_train.append(combo[frame_idx])
            y_train.append(label)

    X_test = []
    y_test = []
    y_test_spikings = []
    for combo, label, spiking in zip(combo_test, label_test, spiking_test):
        for frame_idx in range(combo.shape[0]):
            X_test.append(combo[frame_idx])
            y_test.append(label)
            y_test_spikings.append(spiking)
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    print(f"\nTrain frames: {len(X_train)} (from {len(combo_train)} combos × {sequence_length} frames)")
    print(f"Test frames: {len(X_test)} (from {len(combo_test)} combos × {sequence_length} frames)")
    
    # Create datasets
    train_dataset = SpikingDataset(X_train, y_train, spiking_names)
    test_dataset = SpikingDataset(X_test, y_test, spiking_names,
                                  mean=train_dataset.mean, std=train_dataset.std)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    input_size = X_train.shape[1]
    model = SpatioTemporalCNN(input_size=input_size, num_classes=num_classes)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Learning rate scheduler - reduces LR when validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=False, min_lr=1e-6
    )
    
    # Training loop
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    best_test_acc = 0
    best_epoch = 0
    
    print(f"\nTraining on {len(actual_train_sessions)} sessions...")
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc, _, _ = evaluate(model, test_loader, criterion, device)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
        # Step the scheduler based on validation loss
        scheduler.step(test_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch + 1
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch [{epoch+1}/{epochs}] "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f} | "
                  f"LR: {current_lr:.2e}")
    
    # Final evaluation with probabilities
    _, final_test_acc, test_preds, test_true, test_probs = evaluate(
        model, test_loader, criterion, device, return_probabilities=True
    )
    
    print(f"\nCross-Session Results:")
    print(f"  Best test accuracy: {best_test_acc:.4f} at epoch {best_epoch}")
    print(f"  Final test accuracy: {final_test_acc:.4f}")
    
    # Print average confidence scores per class
    print(f"\n{'='*60}")
    print(f"AVERAGE CONFIDENCE SCORES PER CLASS")
    print(f"{'='*60}")
    for class_idx, spiking_name in enumerate(spiking_names):
        class_mask = (np.array(test_true) == class_idx)
        num_samples = np.sum(class_mask)
        print(f"\nClass {class_idx} ({spiking_name}): {num_samples} samples")
        if num_samples > 0:
            class_probs = np.array(test_probs)[class_mask]
            avg_confidence = np.mean(class_probs[:, class_idx])
            print(f"  Average confidence: {avg_confidence:.4f}")
    
    return {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_accs': train_accs,
        'test_accs': test_accs,
        'best_test_acc': best_test_acc,
        'final_test_acc': final_test_acc,
        'test_preds': test_preds,
        'test_true': test_true,
        'test_probs': test_probs,
        'test_spikings': y_test_spikings,
        'spiking_names': spiking_names,
        'train_sessions': sorted(actual_train_sessions),
        'test_sessions': sorted(actual_test_sessions),
        'sequence_length': sequence_length
    }


def plot_results(results, save_dir='results/coke_spiking', run_name=None):
    """Plot training curves, confusion matrix, and confidence scores for cross-validation or cross-session."""
    os.makedirs(save_dir, exist_ok=True)
    
    if run_name is None:
        run_name = f"coke_spiking"
    
    spiking_names = results['spiking_names']
    
    # Determine if this is cross-validation or cross-session mode
    is_cross_validation = 'n_folds' in results
    
    if is_cross_validation:
        # Cross-validation plotting
        n_folds = results['n_folds']
        fold_results = results['fold_results']
        
        # Create a figure with subplots for each fold plus overall results
        fig = plt.figure(figsize=(24, 14))  # Increased size for better spacing
        
        # Plot 1: Training curves across all folds
        ax1 = plt.subplot(2, 3, 1)
        for fold_idx, fold_result in enumerate(fold_results):
            epochs = range(1, len(fold_result['train_losses']) + 1)
            ax1.plot(epochs, fold_result['train_losses'], alpha=0.3, label=f'Fold {fold_idx+1}')
        
        # Plot average
        avg_train_losses = np.mean([r['train_losses'] for r in fold_results], axis=0)
        ax1.plot(epochs, avg_train_losses, 'b-', linewidth=2, label='Average', alpha=0.8)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Training Loss')
        ax1.set_title('Training Loss (All Folds)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Test loss curves across all folds
        ax2 = plt.subplot(2, 3, 2)
        for fold_idx, fold_result in enumerate(fold_results):
            epochs = range(1, len(fold_result['test_losses']) + 1)
            ax2.plot(epochs, fold_result['test_losses'], alpha=0.3, label=f'Fold {fold_idx+1}')
        
        avg_test_losses = np.mean([r['test_losses'] for r in fold_results], axis=0)
        ax2.plot(epochs, avg_test_losses, 'r-', linewidth=2, label='Average', alpha=0.8)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Test Loss')
        ax2.set_title('Test Loss (All Folds)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Test accuracy curves across all folds
        ax3 = plt.subplot(2, 3, 3)
        for fold_idx, fold_result in enumerate(fold_results):
            epochs = range(1, len(fold_result['test_accs']) + 1)
            ax3.plot(epochs, fold_result['test_accs'], alpha=0.3, label=f'Fold {fold_idx+1}')
        
        avg_test_accs = np.mean([r['test_accs'] for r in fold_results], axis=0)
        ax3.plot(epochs, avg_test_accs, 'g-', linewidth=2, label='Average', alpha=0.8)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Test Accuracy')
        ax3.set_title('Test Accuracy (All Folds)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Use all_test_* for confusion matrix
        test_true = results['all_test_true']
        test_preds = results['all_test_preds']
        test_probs = results['all_test_probs']
        test_spikings = results['all_test_spikings']
        
    else:
        # Cross-session plotting
        fig = plt.figure(figsize=(24, 14))  # Increased size for better spacing
        
        # Plot 1: Training loss
        ax1 = plt.subplot(2, 3, 1)
        epochs = range(1, len(results['train_losses']) + 1)
        ax1.plot(epochs, results['train_losses'], 'b-', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Training Loss')
        ax1.set_title('Training Loss')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Test loss
        ax2 = plt.subplot(2, 3, 2)
        ax2.plot(epochs, results['test_losses'], 'r-', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Test Loss')
        ax2.set_title('Test Loss')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Test accuracy
        ax3 = plt.subplot(2, 3, 3)
        ax3.plot(epochs, results['test_accs'], 'g-', linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Test Accuracy')
        ax3.set_title('Test Accuracy')
        ax3.grid(True, alpha=0.3)
        
        # Use direct test_* from results
        test_true = results['test_true']
        test_preds = results['test_preds']
        test_probs = results['test_probs']
        test_spikings = results['test_spikings']
    
    # Plot 4: Confusion matrix (common for both modes)
    ax4 = plt.subplot(2, 3, 4)
    cm = confusion_matrix(test_true, test_preds)
    # Convert to percentages (normalize by row to show percentage of true class)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Create custom annotations with only percentage
    annotations = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annotations[i, j] = f'{cm_percent[i, j]:.1f}%'

    # Plot heatmap with percentage-based colorbar showing only percentages
    sns.heatmap(cm_percent, annot=annotations, fmt='', cmap='Blues',
                xticklabels=spiking_names, yticklabels=spiking_names, ax=ax4,
                vmin=0, vmax=100,
                cbar_kws={'label': 'Percentage (%)'})
    ax4.set_xlabel('Predicted')
    ax4.set_ylabel('Actual')
    if is_cross_validation:
        ax4.set_title('Confusion Matrix (All Folds)')
    else:
        ax4.set_title('Confusion Matrix')
    
    # Plot 5: Mode-specific visualization
    ax5 = plt.subplot(2, 3, 5)
    if is_cross_validation:
        # Per-fold accuracy comparison
        fold_numbers = [r['fold'] for r in fold_results]
        best_accs = [r['best_test_acc'] for r in fold_results]
        final_accs = [r['final_test_acc'] for r in fold_results]
        
        x = np.arange(len(fold_numbers))
        width = 0.35
        ax5.bar(x - width/2, best_accs, width, label='Best Accuracy', alpha=0.8)
        ax5.bar(x + width/2, final_accs, width, label='Final Accuracy', alpha=0.8)
        ax5.axhline(y=results['avg_best_acc'], color='r', linestyle='--', alpha=0.5, label='Avg Best')
        ax5.axhline(y=results['avg_final_acc'], color='b', linestyle='--', alpha=0.5, label='Avg Final')
        ax5.set_xlabel('Fold')
        ax5.set_ylabel('Accuracy')
        ax5.set_title('Per-Fold Accuracy Comparison')
        ax5.set_xticks(x)
        ax5.set_xticklabels(fold_numbers)
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')
    else:
        # Session information for cross-session (abbreviated to avoid rendering issues)
        ax5.axis('off')
        
        # Shorten session names for display
        def shorten_session(s):
            # Remove common prefix to make shorter
            s = s.replace('session_coke_', '')
            return s
        
        train_sessions_short = [shorten_session(s) for s in results['train_sessions'][:3]]
        test_sessions_short = [shorten_session(s) for s in results['test_sessions'][:3]]
        
        train_suffix = f"\n  ... +{len(results['train_sessions'])-3} more" if len(results['train_sessions']) > 3 else ""
        test_suffix = f"\n  ... +{len(results['test_sessions'])-3} more" if len(results['test_sessions']) > 3 else ""
        
        session_text = f"""Train: {len(results['train_sessions'])} sessions
  {train_sessions_short[0]}
  {train_sessions_short[1] if len(train_sessions_short) > 1 else ''}
  {train_sessions_short[2] if len(train_sessions_short) > 2 else ''}{train_suffix}

Test: {len(results['test_sessions'])} sessions
  {test_sessions_short[0]}
  {test_sessions_short[1] if len(test_sessions_short) > 1 else ''}
  {test_sessions_short[2] if len(test_sessions_short) > 2 else ''}{test_suffix}

Best Acc: {results['best_test_acc']:.4f}
Final Acc: {results['final_test_acc']:.4f}
"""
        ax5.text(0.05, 0.95, session_text, transform=ax5.transAxes, fontsize=7,
                verticalalignment='top', fontfamily='monospace')
    
    # Plot 6: Summary text
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    if is_cross_validation:
        summary_text = f"""Coke Spiking Classification
{n_folds}-Fold Cross-Validation

Model: CNN (per-frame)
Classes: {len(spiking_names)}

Avg Best Accuracy: {results['avg_best_acc']:.4f} ± {results['std_best_acc']:.4f}
Avg Final Accuracy: {results['avg_final_acc']:.4f} ± {results['std_final_acc']:.4f}
Overall Accuracy: {results['overall_acc']:.4f}

Per-fold Best Accuracy:"""
        
        for result in fold_results:
            summary_text += f"\n  Fold {result['fold']}: {result['best_test_acc']:.4f}"
    else:
        summary_text = f"""Coke Spiking Classification
Cross-Session Validation

Model: CNN (per-frame)
Classes: {len(spiking_names)}

Train Sessions: {len(results['train_sessions'])}
Test Sessions: {len(results['test_sessions'])}

Best Test Accuracy: {results['best_test_acc']:.4f}
Final Test Accuracy: {results['final_test_acc']:.4f}"""
    
    # Add per-class metrics (common for both modes)
    summary_text += f"\n\nPer-class performance:"
    class_report = classification_report(test_true, test_preds,
                                       target_names=spiking_names, output_dict=True)
    
    for spiking_name in spiking_names:
        if spiking_name in class_report:
            precision = class_report[spiking_name]['precision']
            recall = class_report[spiking_name]['recall']
            f1 = class_report[spiking_name]['f1-score']
            summary_text += f"\n  {spiking_name}: P={precision:.3f} R={recall:.3f} F1={f1:.3f}"
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=8,
             verticalalignment='top', fontfamily='monospace', wrap=True)
    
    # Adjust layout with padding to avoid rendering issues
    try:
        plt.tight_layout(pad=2.5)
    except Exception as e:
        print(f"Warning: tight_layout failed: {e}")
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.3, hspace=0.3)
    
    # Save figure with mode-specific prefix
    if is_cross_validation:
        fig_path = f'{save_dir}/cv_results_{run_name}.png'
    else:
        fig_path = f'{save_dir}/crosssession_results_{run_name}.png'
    
    # Save with error handling for rendering issues
    try:
        plt.savefig(fig_path, dpi=200, bbox_inches='tight')
        print(f"Saved figure to {fig_path} (200 DPI)")
    except RuntimeError as e:
        print(f"Warning: High DPI save failed ({e}), trying with lower DPI...")
        try:
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            print(f"Saved figure to {fig_path} (150 DPI)")
        except Exception as e2:
            print(f"Warning: Could not save figure with bbox_inches='tight' ({e2}), trying without...")
            plt.savefig(fig_path, dpi=100)
            print(f"Saved figure to {fig_path} (100 DPI, no tight bbox)")
    
    # Try to display the figure, but don't fail if it errors
    try:
        plt.show()
        print("Figure displayed successfully")
    except Exception as e:
        print(f"Note: Could not display figure interactively ({type(e).__name__}), but it was saved successfully")
    finally:
        # Close the figure to free memory
        plt.close(fig)
    
    # Save text summary
    if is_cross_validation:
        summary_path = f'{save_dir}/cv_summary_{run_name}.txt'
    else:
        summary_path = f'{save_dir}/crosssession_summary_{run_name}.txt'
    
    with open(summary_path, 'w') as f:
        f.write(f"Coke Spiking Classification Results\n")
        if is_cross_validation:
            f.write(f"{n_folds}-Fold Cross-Validation\n")
        else:
            f.write(f"Cross-Session Validation\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Model: CNN (per-frame)\n")
        f.write(f"Classes: {len(spiking_names)}\n\n")
        
        if is_cross_validation:
            f.write(f"Average Best Test Accuracy: {results['avg_best_acc']:.4f} ± {results['std_best_acc']:.4f}\n")
            f.write(f"Average Final Test Accuracy: {results['avg_final_acc']:.4f} ± {results['std_final_acc']:.4f}\n")
            f.write(f"Overall Accuracy (all folds): {results['overall_acc']:.4f}\n\n")
            
            f.write("Per-fold Results:\n")
            f.write("-" * 60 + "\n")
            for result in fold_results:
                f.write(f"Fold {result['fold']}: Best={result['best_test_acc']:.4f}, Final={result['final_test_acc']:.4f}\n")
        else:
            f.write(f"Train Sessions ({len(results['train_sessions'])}): {', '.join(results['train_sessions'])}\n")
            f.write(f"Test Sessions ({len(results['test_sessions'])}): {', '.join(results['test_sessions'])}\n\n")
            f.write(f"Best Test Accuracy: {results['best_test_acc']:.4f}\n")
            f.write(f"Final Test Accuracy: {results['final_test_acc']:.4f}\n")
        
        f.write(f"\n{'='*60}\n")
        f.write("Classification Report:\n")
        f.write(f"{'='*60}\n")
        f.write(classification_report(test_true, test_preds, target_names=spiking_names))
        
        # Save confusion matrix
        f.write(f"\n{'='*60}\n")
        f.write("Confusion Matrix:\n")
        f.write(f"{'='*60}\n")
        cm = confusion_matrix(test_true, test_preds)
        f.write(f"\nActual \\ Predicted: {' '.join([f'{name:>10}' for name in spiking_names])}\n")
        for i, name in enumerate(spiking_names):
            f.write(f"{name:>15}: {' '.join([f'{cm[i][j]:>10}' for j in range(len(spiking_names))])}\n")

    # Save raw predictions for later re-plotting
    if is_cross_validation:
        predictions_file = f'{save_dir}/cv_predictions_{run_name}.npz'
        sequence_length = results.get('sequence_length', 30)
        
        np.savez(predictions_file,
                 all_test_preds=results['all_test_preds'],
                 all_test_true=results['all_test_true'],
                 all_test_probs=results['all_test_probs'],
                 all_test_spikings=results['all_test_spikings'],
                 spiking_names=results['spiking_names'],
                 avg_best_acc=results['avg_best_acc'],
                 std_best_acc=results['std_best_acc'],
                 avg_final_acc=results['avg_final_acc'],
                 std_final_acc=results['std_final_acc'],
                 overall_acc=results['overall_acc'],
                 n_folds=results['n_folds'],
                 sequence_length=sequence_length)
    else:
        predictions_file = f'{save_dir}/crosssession_predictions_{run_name}.npz'
        sequence_length = results.get('sequence_length', 30)
        
        np.savez(predictions_file,
                 test_preds=results['test_preds'],
                 test_true=results['test_true'],
                 test_probs=results['test_probs'],
                 test_spikings=results['test_spikings'],
                 spiking_names=results['spiking_names'],
                 best_test_acc=results['best_test_acc'],
                 final_test_acc=results['final_test_acc'],
                 train_sessions=results['train_sessions'],
                 test_sessions=results['test_sessions'],
                 sequence_length=sequence_length)

    print(f"Results saved to {save_dir}/")
    print(f"Raw predictions saved to {predictions_file}")


def plot_sequence_length_comparison(all_results, save_dir='results/coke_spiking', run_name_base=None):
    """
    Plot comparison of model performance across different sequence lengths.
    
    Args:
        all_results: List of tuples (sequence_length, results_dict)
        save_dir: Directory to save the plot
        run_name_base: Base name for the output files
    """
    os.makedirs(save_dir, exist_ok=True)
    
    if run_name_base is None:
        run_name_base = "sequence_length_comparison"
    
    # Sort by sequence length
    all_results = sorted(all_results, key=lambda x: x[0])
    sequence_lengths = [r[0] for r in all_results]
    results_list = [r[1] for r in all_results]
    
    # Get class names from first result
    spiking_names = results_list[0]['spiking_names']
    
    # Determine if cross-validation or cross-session
    is_cross_validation = 'n_folds' in results_list[0]
    
    # Collect per-class accuracies for each sequence length
    per_class_accuracies = {name: [] for name in spiking_names}
    overall_accuracies = []
    
    for seq_len, results in all_results:
        if is_cross_validation:
            test_true = results['all_test_true']
            test_preds = results['all_test_preds']
        else:
            test_true = results['test_true']
            test_preds = results['test_preds']
        
        # Calculate per-class accuracy
        for class_idx, name in enumerate(spiking_names):
            class_mask = (np.array(test_true) == class_idx)
            if np.sum(class_mask) > 0:
                class_preds = np.array(test_preds)[class_mask]
                class_true = np.array(test_true)[class_mask]
                class_acc = accuracy_score(class_true, class_preds)
                per_class_accuracies[name].append(class_acc)
            else:
                per_class_accuracies[name].append(0.0)
        
        # Overall accuracy
        overall_acc = accuracy_score(test_true, test_preds)
        overall_accuracies.append(overall_acc)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot each class
    for name in spiking_names:
        ax.plot(sequence_lengths, per_class_accuracies[name], 
                marker='o', linewidth=2, markersize=8, label=name, alpha=0.7)
    
    # Plot overall accuracy
    ax.plot(sequence_lengths, overall_accuracies, 
            marker='s', linewidth=3, markersize=10, label='Overall Average', 
            color='black', linestyle='--', alpha=0.9)
    
    ax.set_xlabel('Sequence Length (frames)', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Classification Accuracy vs Sequence Length', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Set x-axis to show all sequence lengths
    ax.set_xticks(sequence_lengths)
    ax.set_xticklabels([str(s) for s in sequence_lengths])
    
    # Set y-axis range from 0 to 1
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    
    # Save figure
    fig_path = f'{save_dir}/sequence_length_comparison_{run_name_base}.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\nSequence length comparison plot saved to {fig_path}")
    
    # Try to display
    try:
        plt.show()
    except Exception as e:
        print(f"Note: Could not display figure interactively ({type(e).__name__})")
    finally:
        plt.close(fig)
    
    # Save detailed text summary
    summary_path = f'{save_dir}/sequence_length_comparison_{run_name_base}.txt'
    with open(summary_path, 'w') as f:
        f.write("Sequence Length Comparison Results\n")
        f.write("="*60 + "\n\n")
        
        # Write table header
        f.write(f"{'Seq Len':<10}")
        for name in spiking_names:
            f.write(f"{name:<15}")
        f.write(f"{'Overall':<15}\n")
        f.write("-"*60 + "\n")
        
        # Write data rows
        for i, seq_len in enumerate(sequence_lengths):
            f.write(f"{seq_len:<10}")
            for name in spiking_names:
                f.write(f"{per_class_accuracies[name][i]:<15.4f}")
            f.write(f"{overall_accuracies[i]:<15.4f}\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("\nSummary:\n")
        f.write(f"Best overall accuracy: {max(overall_accuracies):.4f} at sequence length {sequence_lengths[np.argmax(overall_accuracies)]}\n")
        f.write(f"Worst overall accuracy: {min(overall_accuracies):.4f} at sequence length {sequence_lengths[np.argmin(overall_accuracies)]}\n")
        
        f.write("\nPer-class best accuracies:\n")
        for name in spiking_names:
            best_acc = max(per_class_accuracies[name])
            best_seq = sequence_lengths[np.argmax(per_class_accuracies[name])]
            f.write(f"  {name}: {best_acc:.4f} at sequence length {best_seq}\n")
    
    print(f"Detailed comparison summary saved to {summary_path}")
    
    return fig_path, summary_path


def main():
    """Main training function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train coke spiking classifier (per-frame)')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for training (default: 16)')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs (default: 10)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--n-regions', type=int, default=1,
                       help='Number of regions to group and average as one sample (default: 1)')
    parser.add_argument('--classes', type=str, default=None,
                       help='Comma-separated list of spiking classes to use. Can be combined with --binary (default: all)')
    parser.add_argument('--binary', action='store_true',
                       help='Binary classification: unadulterated vs spiked. Can be combined with --classes to select specific types (default: False)')
    parser.add_argument('--use-all-combinations', action='store_true',
                       help='Use all C(N,n) combinations (may cause data leakage). Default: False (use non-overlapping sampling)')
    parser.add_argument('--cross-session', action='store_true',
                       help='Use cross-session validation instead of cross-validation (default: False)')
    parser.add_argument('--test-sessions', type=str, default=None,
                       help='Comma-separated list of session names to use for testing in cross-session mode. If not provided, 20%% of sessions will be automatically selected.')
    parser.add_argument('--day-suffix', type=str, default=None,
                       help='Day identifier suffix (e.g., "0109" for Jan 9 data). If not specified, uses original data (no date suffix).')
    parser.add_argument('--sequence-length', type=str, default='50',
                       help='Number of frames to extract from each sample (default: 50). Sequences will be padded or cropped to this length. Can provide comma-separated list (e.g., "50,20,10,5") to run multiple experiments.')
    parser.add_argument('--save-model', action='store_true',
                       help='Save the best model (PyTorch .pth and ONNX .onnx) for deployment (default: False)')
    parser.add_argument('--model-dir', type=str, default='models',
                       help='Directory to save trained models (default: models/)')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Parse sequence lengths (can be comma-separated)
    sequence_lengths_str = args.sequence_length.strip()
    if ',' in sequence_lengths_str:
        sequence_lengths = [int(s.strip()) for s in sequence_lengths_str.split(',')]
        multiple_sequence_lengths = True
        print(f"\nMultiple sequence length experiment: {sequence_lengths}")
    else:
        sequence_lengths = [int(sequence_lengths_str)]
        multiple_sequence_lengths = False
    
    # Parse selected classes
    if args.classes is not None:
        selected_classes = [c.strip() for c in args.classes.split(',')]
        # Validate classes
        invalid_classes = [c for c in selected_classes if c not in ALL_SPIKINGS]
        if invalid_classes:
            print(f"Error: Invalid classes: {invalid_classes}")
            print(f"Available classes: {ALL_SPIKINGS}")
            return
        if args.binary:
            print(f"Binary mode with selected classes: {selected_classes}")
            print("Classes will be grouped into unadulterated vs spiked")
    else:
        selected_classes = None
    
    # Parse test sessions if provided
    test_sessions = None
    if args.test_sessions is not None:
        test_sessions = [s.strip() for s in args.test_sessions.split(',')]
    
    # Store all results if running multiple sequence lengths
    all_seq_len_results = []
    
    # Run training for each sequence length
    for seq_idx, sequence_length in enumerate(sequence_lengths):
        if multiple_sequence_lengths:
            print(f"\n{'='*70}")
            print(f"EXPERIMENT {seq_idx + 1}/{len(sequence_lengths)}: Sequence Length = {sequence_length}")
            print(f"{'='*70}\n")
        
        print("Loading coke spiking data...")
        if args.day_suffix:
            print(f"Using data from day: {args.day_suffix}")
        else:
            print(f"Using original data (no date suffix)")
        print(f"Sequence length: {sequence_length} frames per sample")
        print(f"Sampling mode: {'All combinations (may have data leakage)' if args.use_all_combinations else 'Non-overlapping (no data leakage)'}")
        data = load_coke_spiking_data(n_regions=args.n_regions, 
                                       sequence_length=sequence_length,
                                       use_all_combinations=args.use_all_combinations,
                                       selected_classes=selected_classes,
                                       binary=args.binary,
                                       day_suffix=args.day_suffix)

        if len(data['spiking_names']) < 2:
            print("Error: Need at least 2 classes for classification")
            return
        
        # Choose training mode
        if args.cross_session:
            print(f"\nStarting cross-session validation with CNN model (per-frame)...")
            results = train_cross_session(data, test_sessions=test_sessions, 
                                           epochs=args.epochs, batch_size=args.batch_size)
            if results is None:
                return
            mode_suffix = "crosssession"
        else:
            # Create run name for model saving
            combo_mode = "allcombos" if args.use_all_combinations else "nonoverlap"
            class_mode = "binary" if args.binary else "multiclass"
            day_suffix_str = f"_day{args.day_suffix}" if args.day_suffix else ""
            seq_len_str = f"_seq{sequence_length}" if sequence_length != 50 else ""

            # Add class selection to model name if specified
            if selected_classes is not None:
                # Create a short identifier from selected classes
                class_suffix = "_".join(sorted(selected_classes))
                model_name = f"coke_spiking_{class_mode}_n{args.n_regions}_{combo_mode}_cv5{day_suffix_str}{seq_len_str}_{class_suffix}"
            else:
                model_name = f"coke_spiking_{class_mode}_n{args.n_regions}_{combo_mode}_cv5{day_suffix_str}{seq_len_str}"
            
            # Set model save path if saving is enabled
            model_save_path = None
            if args.save_model:
                model_save_path = os.path.join(args.model_dir, model_name)
            
            print(f"\nStarting 5-fold cross-validation with CNN model (per-frame)...")
            results = train_cross_validation(data, n_folds=5, epochs=args.epochs, batch_size=args.batch_size,
                                            save_model=args.save_model, model_save_path=model_save_path)
            mode_suffix = "cv5"

        # Create run name for results
        combo_mode = "allcombos" if args.use_all_combinations else "nonoverlap"
        class_mode = "binary" if args.binary else "multiclass"
        day_suffix_str = f"_day{args.day_suffix}" if args.day_suffix else ""
        seq_len_str = f"_seq{sequence_length}" if sequence_length != 50 else ""

        # Add class selection to run name if specified
        if selected_classes is not None:
            # Create a short identifier from selected classes
            class_suffix = "_".join(sorted(selected_classes))
            run_name = f"coke_spiking_{class_mode}_n{args.n_regions}_{combo_mode}_{mode_suffix}{day_suffix_str}{seq_len_str}_{class_suffix}"
        else:
            run_name = f"coke_spiking_{class_mode}_n{args.n_regions}_{combo_mode}_{mode_suffix}{day_suffix_str}{seq_len_str}"
        
        print("\nPlotting results...")
        plot_results(results, run_name=run_name)
        
        # Store results for comparison plot
        if multiple_sequence_lengths:
            all_seq_len_results.append((sequence_length, results))
        
        print(f"\nTraining complete for sequence length {sequence_length}!")
    
    # If multiple sequence lengths, create comparison plot
    if multiple_sequence_lengths:
        print(f"\n{'='*70}")
        print("GENERATING SEQUENCE LENGTH COMPARISON PLOT")
        print(f"{'='*70}\n")
        
        # Create base run name without sequence length
        combo_mode = "allcombos" if args.use_all_combinations else "nonoverlap"
        class_mode = "binary" if args.binary else "multiclass"
        day_suffix_str = f"day{args.day_suffix}" if args.day_suffix else "default"
        
        if selected_classes is not None:
            class_suffix = "_".join(sorted(selected_classes))
            run_name_base = f"{class_mode}_n{args.n_regions}_{combo_mode}_{mode_suffix}_{day_suffix_str}_{class_suffix}"
        else:
            run_name_base = f"{class_mode}_n{args.n_regions}_{combo_mode}_{mode_suffix}_{day_suffix_str}"
        
        plot_sequence_length_comparison(all_seq_len_results, run_name_base=run_name_base)
        
        # Save summary for cross-task visualization
        os.makedirs('results/sequence_length_comparison', exist_ok=True)
        summary_data = {
            'task': 'coke_spiking',
            'sequence_lengths': [r[0] for r in all_seq_len_results],
            'overall_accuracies': [r[1]['overall_acc'] for r in all_seq_len_results]
        }
        
        import json
        summary_file = 'results/sequence_length_comparison/coke_spiking.json'
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        print(f"Sequence length comparison data saved to {summary_file}")
        
        print(f"\nAll experiments complete! Results saved to results/")
    else:
        print(f"\nTraining complete! Results saved to results/")

if __name__ == "__main__":
    main()
