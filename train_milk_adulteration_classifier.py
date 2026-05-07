#!/usr/bin/env python3
"""
Train a CNN model for milk adulteration classification (per-frame basis).

The model uses:
- CNN layers to extract spatial features from each frame
- Each frame is treated as an independent sample (no temporal modeling)
- Train/test split for evaluation
- Region batching: randomly group n regions, average them, then extract each frame as a sample
"""
import numpy as np
import matplotlib.pyplot as plt
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
import seaborn as sns

# Check for help flags early
if '--help' in sys.argv or '-h' in sys.argv:
    print("""usage: train_milk_adulteration_classifier.py [-h] [--batch-size BATCH_SIZE]
                                          [--seed SEED] [--n-regions N_REGIONS]
                                          [--classes CLASSES]
                                          [--binary]

Train milk adulteration classifier with CNN (per-frame)

optional arguments:
  -h, --help            show this help message and exit
  --batch-size BATCH_SIZE
                        Batch size for training (default: 16)
  --seed SEED           Random seed for reproducibility (default: 42)
  --n-regions N_REGIONS
                        Number of regions to group and average as one sample (default: 6)
  --classes CLASSES     Comma-separated list of adulteration classes to use.
                        Available: unadulterated, detergent, starch, salt, tylenol, water
                        (default: all classes)
  --binary              Binary classification: adulterated vs unadulterated
                        (default: False - multi-class classification)
  --use-all-combinations
                        Use all C(N,n) combinations (may cause data leakage).
                        Default: False (use non-overlapping random sampling)
  --info                Show detailed information about training
                        configurations""")
    sys.exit(0)

if '--info' in sys.argv:
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║     MILK ADULTERATION CLASSIFIER TRAINING (PER-FRAME)                ║
╚══════════════════════════════════════════════════════════════════════╝

This script trains a CNN model to classify milk adulteration from
individual sensor frames (no temporal modeling).

──────────────────────────────────────────────────────────────────────
DATA FORMAT
──────────────────────────────────────────────────────────────────────
Expects region files in 'regions/' directory:
- session_milk_unadulterated_regions.npz
- session_milk_detergent_regions.npz
- session_milk_starch_regions.npz
- session_milk_salt_regions.npz
- session_milk_tylenol_regions.npz
- session_milk_water_regions.npz

──────────────────────────────────────────────────────────────────────
CLASSIFICATION MODES
──────────────────────────────────────────────────────────────────────
1. Multi-class classification (default):
   - Classify each adulteration type separately
   - Can select specific classes to include

2. Binary classification (--binary):
   - Classify as adulterated vs unadulterated
   - All adulteration types grouped into one class

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
# Train multi-class with all adulteration types (default)
python3 train_milk_adulteration_classifier.py

# Train binary classification (adulterated vs unadulterated)
python3 train_milk_adulteration_classifier.py --binary

# Train only on specific classes
python3 train_milk_adulteration_classifier.py --classes unadulterated,detergent,water

# Train with different number of regions
python3 train_milk_adulteration_classifier.py --n-regions 4

# Train with all combinations (may have data leakage)
python3 train_milk_adulteration_classifier.py --n-regions 2 --use-all-combinations

# Sequence length experiments
python3 train_milk_adulteration_classifier.py --sequence-length 50,20,10,5,2,1 --epochs 10 --binary --classes unadulterated,detergent,salt25
══════════════════════════════════════════════════════════════════════
""")
    sys.exit(0)

REGIONS_DIR = 'regions'

# All available adulteration types
ALL_ADULTERATIONS = ['unadulterated', 'detergent', 'starch', 'salt', 'tylenol', 'water', 'water20', 'salt25']

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

class AdulterationDataset(Dataset):
    """Dataset for milk adulteration classification from temporal patches."""

    def __init__(self, patches, labels, adulterations, normalize=True, mean=None, std=None):
        self.patches = torch.FloatTensor(patches)
        self.labels = torch.LongTensor(labels)
        self.adulterations = adulterations
        
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

    def __init__(self, input_size=7, num_classes=6, hidden_dim=64, lstm_layers=2):
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

    def __init__(self, input_size=7, num_classes=6):
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

def load_milk_adulteration_data(patches_dir="regions", sequence_length=50, n_regions=6, 
                                use_all_combinations=False, selected_classes=None, binary=False):
    """
    Load milk adulteration patch data and create samples by batching regions.
    
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
        List of adulteration types to include. If None, use all available.
    binary : bool
        If True, binary classification (adulterated vs unadulterated).
        If False, multi-class classification.
    
    Returns:
    --------
    data : dict
        Contains 'combinations', 'combo_labels', 'combo_adulterations', 
        'adulteration_names', 'label_to_adulteration', 'sequence_length'
    """
    patches_dir = Path(patches_dir)
    
    # Determine which classes to load
    if selected_classes is None:
        adulterations_to_load = ALL_ADULTERATIONS
    else:
        adulterations_to_load = selected_classes
    
    print(f"Loading adulteration types: {adulterations_to_load}")
    if binary:
        print("Binary classification mode: adulterated vs unadulterated")
    else:
        print("Multi-class classification mode")
    
    # Group by adulteration type
    adulteration_data = defaultdict(list)
    original_lengths = []
    
    for adulteration in adulterations_to_load:
        # Find all files matching this adulteration type (handles multiple sessions)
        # Match both patterns: with and without session number
        # e.g., session_milk_unadulterated_regions.npz AND session_milk_unadulterated_2_regions.npz
        pattern1 = f'session_milk_{adulteration}_regions.npz'  # without number
        pattern2 = f'session_milk_{adulteration}_*_regions.npz'  # with number
        
        matching_files = []
        # Check for file without number
        file_without_number = patches_dir / pattern1
        if file_without_number.exists():
            matching_files.append(file_without_number)
        
        # Check for files with numbers
        matching_files.extend([f for f in patches_dir.glob(pattern2) if f != file_without_number])
        matching_files = sorted(matching_files)
        
        if not matching_files:
            print(f"Warning: No files found for {adulteration}, skipping")
            continue
        
        print(f"  Found {len(matching_files)} file(s) for {adulteration}")
        
        # Load data from all matching files
        for file in matching_files:
            region_patches = load_region_data(file)
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
                adulteration_data[adulteration].append(patch)
    
    if not adulteration_data:
        raise ValueError("No data loaded! Check that the region files exist.")
    
    print(f"Loaded data for: {sorted(adulteration_data.keys())}")
    print(f"Original sequence lengths: min={min(original_lengths)}, max={max(original_lengths)}, mean={np.mean(original_lengths):.1f}")
    
    # Determine common spatial dimensions (use the maximum dimensions found)
    all_spatial_dims = []
    for adulteration, patches in adulteration_data.items():
        for patch in patches:
            all_spatial_dims.append(patch.shape[1:])  # (H, W)
    
    max_h = max(dim[0] for dim in all_spatial_dims)
    max_w = max(dim[1] for dim in all_spatial_dims)
    print(f"Standardizing spatial dimensions to: ({max_h}, {max_w})")
    
    # Resize all patches to common dimensions
    for adulteration in adulteration_data.keys():
        resized_patches = []
        for patch in adulteration_data[adulteration]:
            if patch.shape[1] != max_h or patch.shape[2] != max_w:
                # Resize spatial dimensions
                zoom_factors = (1.0, max_h / patch.shape[1], max_w / patch.shape[2])
                resized_patch = zoom(patch, zoom_factors, order=1)
                resized_patches.append(resized_patch)
            else:
                resized_patches.append(patch)
        adulteration_data[adulteration] = resized_patches
    
    # Create labels
    if binary:
        # Binary classification: unadulterated (0) vs adulterated (1)
        adulteration_names = ['unadulterated', 'adulterated']
        label_to_adulteration = {0: 'unadulterated', 1: 'adulterated'}
        
        # Map each adulteration type to binary label
        def get_binary_label(adulteration):
            return 0 if adulteration == 'unadulterated' else 1
    else:
        # Multi-class classification
        adulteration_names = sorted(adulteration_data.keys())
        label_to_adulteration = {i: adulteration for i, adulteration in enumerate(adulteration_names)}
        adulteration_to_label = {adulteration: i for i, adulteration in enumerate(adulteration_names)}
    
    # Now create samples by generating combinations of n_regions
    # Store combinations first, then split, then extract frames to avoid data leakage
    all_combinations = []
    all_combo_labels = []
    all_combo_adulterations = []
    
    for adulteration, patches in adulteration_data.items():
        print(f"\n{adulteration}: {len(patches)} regions")
        
        # Check if we have enough regions
        if len(patches) < n_regions:
            print(f"  Warning: Only {len(patches)} regions available, need at least {n_regions}")
            print(f"  Skipping this adulteration type")
            continue
        
        if binary:
            label = get_binary_label(adulteration)
        else:
            label = adulteration_to_label[adulteration]
        
        if use_all_combinations:
            # Generate all combinations of n_regions from available patches
            all_combos = list(combinations(range(len(patches)), n_regions))
            print(f"  Using all combinations: C({len(patches)}, {n_regions}) = {len(all_combos)}")
            print(f"  WARNING: This may cause data leakage due to overlapping regions!")
            
            # Create averaged sequence for each combination
            for combo_indices in all_combos:
                # Get the patches for this combination
                combo_patches = [patches[i] for i in combo_indices]
                
                # Average the patches
                combo_array = np.array(combo_patches)  # (n_regions, seq_len, H, W)
                averaged_sample = np.mean(combo_array, axis=0)  # (seq_len, H, W)
                
                # Store the combination
                all_combinations.append(averaged_sample)
                all_combo_labels.append(label)
                all_combo_adulterations.append(adulteration)
            
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
                
                # Average the patches
                combo_array = np.array(combo_patches)  # (n_regions, seq_len, H, W)
                averaged_sample = np.mean(combo_array, axis=0)  # (seq_len, H, W)
                
                # Store the combination
                all_combinations.append(averaged_sample)
                all_combo_labels.append(label)
                all_combo_adulterations.append(adulteration)
            
            print(f"  Created {num_samples} non-overlapping samples (will expand to {num_samples * sequence_length} frames)")
    
    # Convert combinations to numpy arrays
    all_combinations = np.array(all_combinations)  # (num_combos, seq_len, H, W)
    all_combo_labels = np.array(all_combo_labels)
    all_combo_adulterations = np.array(all_combo_adulterations)
    
    print(f"\nCombination summary:")
    print(f"  Total combinations: {len(all_combinations)}")
    print(f"  Combination shape: {all_combinations.shape}")
    print(f"  Number of classes: {len(adulteration_names)}")
    print(f"  Combinations per class:")
    for i, name in enumerate(adulteration_names):
        count = np.sum(all_combo_labels == i)
        print(f"    {name}: {count} combinations")
    
    return {
        'combinations': all_combinations,  # (num_combos, seq_len, H, W)
        'combo_labels': all_combo_labels,
        'combo_adulterations': all_combo_adulterations,
        'adulteration_names': adulteration_names,
        'label_to_adulteration': label_to_adulteration,
        'sequence_length': sequence_length,
        'binary': binary
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

def train_cross_validation(data, n_folds=5, epochs=100, batch_size=16, lr=0.001):
    """Train and evaluate model with k-fold cross-validation."""
    combinations = data['combinations']  # (num_combos, seq_len, H, W)
    combo_labels = data['combo_labels']
    combo_adulterations = data['combo_adulterations']  # Original adulteration types before binary conversion
    adulteration_names = data['adulteration_names']
    sequence_length = data['sequence_length']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set up k-fold cross-validation at the COMBINATION level to avoid data leakage
    print(f"\n=== {n_folds}-Fold Cross-Validation (no data leakage) ===")
    
    num_classes = len(adulteration_names)
    
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
    all_test_adulterations = []  # Store original adulteration types

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx + 1}/{n_folds}")
        print(f"{'='*60}")
        
        # Split combinations
        combo_train = combinations[train_idx]
        combo_test = combinations[test_idx]
        label_train = combo_labels[train_idx]
        label_test = combo_labels[test_idx]
        adulteration_train = combo_adulterations[train_idx]
        adulteration_test = combo_adulterations[test_idx]

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
        y_test_adulterations = []
        for combo, label, adulteration in zip(combo_test, label_test, adulteration_test):
            for frame_idx in range(combo.shape[0]):
                X_test.append(combo[frame_idx])
                y_test.append(label)
                y_test_adulterations.append(adulteration)
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        
        print(f"Train frames: {len(X_train)} (from {len(combo_train)} combos × {sequence_length} frames)")
        print(f"Test frames: {len(X_test)} (from {len(combo_test)} combos × {sequence_length} frames)")
        
        # Create datasets
        train_dataset = AdulterationDataset(X_train, y_train, adulteration_names)
        test_dataset = AdulterationDataset(X_test, y_test, adulteration_names,
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
        
        # Training loop
        train_losses = []
        test_losses = []
        train_accs = []
        test_accs = []
        best_test_acc = 0
        best_epoch = 0
        
        print(f"\nTraining fold {fold_idx + 1}...")
        
        for epoch in range(epochs):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            test_loss, test_acc, _, _ = evaluate(model, test_loader, criterion, device)
            
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_epoch = epoch + 1
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch [{epoch+1}/{epochs}] "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                      f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        
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
        all_test_adulterations.extend(y_test_adulterations)
    
    # Compute overall statistics
    all_test_preds = np.array(all_test_preds)
    all_test_true = np.array(all_test_true)
    all_test_probs = np.array(all_test_probs)
    all_test_adulterations = np.array(all_test_adulterations)
    
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
    
    # Print average confidence scores per class
    print(f"\n{'='*60}")
    print(f"AVERAGE CONFIDENCE SCORES PER CLASS (ALL FOLDS)")
    print(f"{'='*60}")
    for class_idx, adulteration_name in enumerate(adulteration_names):
        class_mask = (all_test_true == class_idx)
        num_samples = np.sum(class_mask)
        print(f"\nClass {class_idx} ({adulteration_name}): {num_samples} samples")
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
        'all_test_adulterations': all_test_adulterations,
        'adulteration_names': adulteration_names,
        'n_folds': n_folds,
        'sequence_length': sequence_length
    }

def plot_results(results, save_dir='results/milk_adulteration', run_name=None):
    """Plot training curves, confusion matrix, and confidence scores for cross-validation."""
    os.makedirs(save_dir, exist_ok=True)
    
    if run_name is None:
        run_name = f"milk_adulteration"
    
    n_folds = results['n_folds']
    fold_results = results['fold_results']
    adulteration_names = results['adulteration_names']
    
    # Create a figure with subplots for each fold plus overall results
    fig = plt.figure(figsize=(20, 12))
    
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
    
    # Plot 4: Overall confusion matrix (all folds combined)
    ax4 = plt.subplot(2, 3, 4)
    cm = confusion_matrix(results['all_test_true'], results['all_test_preds'])
    # Convert to percentages (normalize by row to show percentage of true class)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Create custom annotations with both count and percentage
    annotations = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annotations[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'

    # Plot heatmap with percentage-based colorbar but showing both values
    sns.heatmap(cm_percent, annot=annotations, fmt='', cmap='Blues',
                xticklabels=adulteration_names, yticklabels=adulteration_names, ax=ax4,
                vmin=0, vmax=100,
                cbar_kws={'label': 'Percentage (%)'})
    ax4.set_xlabel('Predicted')
    ax4.set_ylabel('Actual')
    ax4.set_title('Confusion Matrix (All Folds)')
    
    # Plot 5: Per-fold accuracy comparison
    ax5 = plt.subplot(2, 3, 5)
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
    
    # Plot 6: Summary text
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    summary_text = f"""
    Milk Adulteration Classification Results
    {n_folds}-Fold Cross-Validation
    
    Model: CNN (per-frame)
    Classes: {len(adulteration_names)}
    
    Average Best Accuracy: {results['avg_best_acc']:.4f} ± {results['std_best_acc']:.4f}
    Average Final Accuracy: {results['avg_final_acc']:.4f} ± {results['std_final_acc']:.4f}
    Overall Accuracy: {results['overall_acc']:.4f}
    
    Per-fold Best Accuracy:
    """
    
    for result in fold_results:
        summary_text += f"\n    Fold {result['fold']}: {result['best_test_acc']:.4f}"
    
    # Add per-class metrics
    summary_text += f"\n\n    Per-class performance (all folds):"
    class_report = classification_report(results['all_test_true'], results['all_test_preds'],
                                       target_names=adulteration_names, output_dict=True)
    
    for adulteration_name in adulteration_names:
        if adulteration_name in class_report:
            precision = class_report[adulteration_name]['precision']
            recall = class_report[adulteration_name]['recall']
            f1 = class_report[adulteration_name]['f1-score']
            summary_text += f"\n    {adulteration_name}: P={precision:.3f} R={recall:.3f} F1={f1:.3f}"
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/cv_results_{run_name}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save text summary
    with open(f'{save_dir}/cv_summary_{run_name}.txt', 'w') as f:
        f.write(f"Milk Adulteration Classification Results\n")
        f.write(f"{n_folds}-Fold Cross-Validation\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Model: CNN (per-frame)\n")
        f.write(f"Classes: {len(adulteration_names)}\n\n")
        f.write(f"Average Best Test Accuracy: {results['avg_best_acc']:.4f} ± {results['std_best_acc']:.4f}\n")
        f.write(f"Average Final Test Accuracy: {results['avg_final_acc']:.4f} ± {results['std_final_acc']:.4f}\n")
        f.write(f"Overall Accuracy (all folds): {results['overall_acc']:.4f}\n\n")
        
        f.write("Per-fold Results:\n")
        f.write("-" * 60 + "\n")
        for result in fold_results:
            f.write(f"Fold {result['fold']}: Best={result['best_test_acc']:.4f}, Final={result['final_test_acc']:.4f}\n")
        
        f.write(f"\n{'='*60}\n")
        f.write("Classification Report (All Folds Combined):\n")
        f.write(f"{'='*60}\n")
        f.write(classification_report(results['all_test_true'], results['all_test_preds'],
                                     target_names=adulteration_names))
        
        # Save confusion matrix
        f.write(f"\n{'='*60}\n")
        f.write("Confusion Matrix (All Folds Combined):\n")
        f.write(f"{'='*60}\n")
        cm = confusion_matrix(results['all_test_true'], results['all_test_preds'])
        f.write(f"\nActual \\ Predicted: {' '.join([f'{name:>10}' for name in adulteration_names])}\n")
        for i, name in enumerate(adulteration_names):
            f.write(f"{name:>15}: {' '.join([f'{cm[i][j]:>10}' for j in range(len(adulteration_names))])}\n")

    # Save raw predictions for later re-plotting
    predictions_file = f'{save_dir}/cv_predictions_{run_name}.npz'

    # Get sequence_length from results
    sequence_length = results.get('sequence_length', 50)  # default to 50 for milk

    np.savez(predictions_file,
             all_test_preds=results['all_test_preds'],
             all_test_true=results['all_test_true'],
             all_test_probs=results['all_test_probs'],
             all_test_adulterations=results['all_test_adulterations'],
             adulteration_names=results['adulteration_names'],
             avg_best_acc=results['avg_best_acc'],
             std_best_acc=results['std_best_acc'],
             avg_final_acc=results['avg_final_acc'],
             std_final_acc=results['std_final_acc'],
             overall_acc=results['overall_acc'],
             n_folds=results['n_folds'],
             sequence_length=sequence_length)

    print(f"Results saved to {save_dir}/")
    print(f"Raw predictions saved to {predictions_file}")

def main():
    """Main training function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train milk adulteration classifier (per-frame)')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for training (default: 16)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--n-regions', type=int, default=1,
                       help='Number of regions to group and average as one sample (default: 1)')
    parser.add_argument('--classes', type=str, default=None,
                       help='Comma-separated list of adulteration classes to use (default: all)')
    parser.add_argument('--binary', action='store_true',
                       help='Binary classification: adulterated vs unadulterated (default: False)')
    parser.add_argument('--use-all-combinations', action='store_true',
                       help='Use all C(N,n) combinations (may cause data leakage). Default: False (use non-overlapping sampling)')
    parser.add_argument('--sequence-length', type=str, default='50',
                       help='Number of frames to extract from each sample (default: 50). Can provide comma-separated list (e.g., "50,20,10,5").')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    
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
        invalid_classes = [c for c in selected_classes if c not in ALL_ADULTERATIONS]
        if invalid_classes:
            print(f"Error: Invalid classes: {invalid_classes}")
            print(f"Available classes: {ALL_ADULTERATIONS}")
            return
    else:
        selected_classes = None
    
    # Store all results if running multiple sequence lengths
    all_seq_len_results = []
    
    # Run training for each sequence length
    for seq_idx, sequence_length in enumerate(sequence_lengths):
        if multiple_sequence_lengths:
            print(f"\n{'='*70}")
            print(f"EXPERIMENT {seq_idx + 1}/{len(sequence_lengths)}: Sequence Length = {sequence_length}")
            print(f"{'='*70}\n")
    
        print("Loading milk adulteration data...")
        print(f"Sequence length: {sequence_length} frames per sample")
        print(f"Sampling mode: {'All combinations (may have data leakage)' if args.use_all_combinations else 'Non-overlapping (no data leakage)'}")
        data = load_milk_adulteration_data(n_regions=args.n_regions,
                                           sequence_length=sequence_length, 
                                           use_all_combinations=args.use_all_combinations,
                                           selected_classes=selected_classes,
                                           binary=args.binary)
        
        if len(data['adulteration_names']) < 2:
            print("Error: Need at least 2 classes for classification")
            return
        
        print(f"\nStarting 5-fold cross-validation with CNN model (per-frame)...")
        results = train_cross_validation(data, n_folds=5, epochs=args.epochs, batch_size=args.batch_size)
    
        # Create run name
        combo_mode = "allcombos" if args.use_all_combinations else "nonoverlap"
        class_mode = "binary" if args.binary else "multiclass"
        seq_len_str = f"_seq{sequence_length}" if sequence_length != 50 else ""
        run_name = f"milk_{class_mode}_n{args.n_regions}_{combo_mode}{seq_len_str}"
        
        print("\nPlotting results...")
        plot_results(results, run_name=run_name)
        
        # Store results for comparison plot
        if multiple_sequence_lengths:
            all_seq_len_results.append((sequence_length, results))
        
        print(f"\nTraining complete for sequence length {sequence_length}!")
    
    # If multiple sequence lengths, save summary for visualization
    if multiple_sequence_lengths:
        print(f"\n{'='*70}")
        print("SAVING SEQUENCE LENGTH COMPARISON DATA")
        print(f"{'='*70}\n")
        
        os.makedirs('results/sequence_length_comparison', exist_ok=True)
        summary_data = {
            'task': 'milk_adulteration',
            'sequence_lengths': [r[0] for r in all_seq_len_results],
            'overall_accuracies': [r[1]['overall_acc'] for r in all_seq_len_results]
        }
        
        import json
        summary_file = 'results/sequence_length_comparison/milk_adulteration.json'
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        print(f"Sequence length comparison data saved to {summary_file}")
        print(f"\nAll experiments complete! Results saved to results/")
    else:
        print(f"\nTraining complete! Results saved to results/")

if __name__ == "__main__":
    main()
