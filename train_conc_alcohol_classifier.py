#!/usr/bin/env python3
"""
Train a CNN model for alcohol concentration classification (per-frame basis).

The model uses:
- CNN layers to extract spatial features from each frame
- Each frame is treated as an independent sample (no temporal modeling)
- Train/test split for evaluation
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from collections import defaultdict
from scipy.ndimage import zoom
from itertools import combinations
import random
import argparse
import os
import sys
import re

# Check for help flags early
if '--help' in sys.argv or '-h' in sys.argv:
    print("""usage: train_conc_alcohol_classifier.py [-h] [--batch-size BATCH_SIZE]
                                          [--seed SEED] [--n-regions N_REGIONS]

Train alcohol concentration classifier with CNN (per-frame)

optional arguments:
  -h, --help            show this help message and exit
  --batch-size BATCH_SIZE
                        Batch size for training (default: 16)
  --seed SEED           Random seed for reproducibility (default: 42)
  --n-regions N_REGIONS
                        Number of regions to group and average as one sample (default: 6)
  --use-all-combinations
                        Use all C(N,n) combinations (may cause data leakage).
                        Default: False (use non-overlapping random sampling)
  --info                Show detailed information about training
                        configurations""")
    sys.exit(0)

if '--info' in sys.argv:
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║     ALCOHOL CONCENTRATION CLASSIFIER TRAINING (PER-FRAME)            ║
╚══════════════════════════════════════════════════════════════════════╝

This script trains a CNN model to classify alcohol concentrations from
individual sensor frames (no temporal modeling).

──────────────────────────────────────────────────────────────────────
DATA FORMAT
──────────────────────────────────────────────────────────────────────
Expects region files in 'regions/' directory:
- session_conc_alcohol_20_regions.npz (20% concentration)
- session_conc_alcohol_40_regions.npz (40% concentration)
- session_conc_alcohol_60_regions.npz (60% concentration)
- session_conc_alcohol_80_regions.npz (80% concentration)
- session_conc_alcohol_100_regions.npz (100% concentration)
- session_conc_0_regions.npz (0% concentration, shared with other experiments)

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
# Train with default settings (n=6 regions per sample)
python3 train_conc_alcohol_classifier.py

# Train with different number of regions (non-overlapping)
python3 train_conc_alcohol_classifier.py --n-regions 4

# Train with all combinations (may have data leakage)
python3 train_conc_alcohol_classifier.py --n-regions 2 --use-all-combinations

══════════════════════════════════════════════════════════════════════
""")
    sys.exit(0)

REGIONS_DIR = 'regions'

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

class ConcentrationDataset(Dataset):
    """Dataset for concentration classification from temporal patches."""

    def __init__(self, patches, labels, concentrations, normalize=True, mean=None, std=None):
        self.patches = torch.FloatTensor(patches)
        self.labels = torch.LongTensor(labels)
        self.concentrations = concentrations
        
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

def load_alcohol_concentration_data(patches_dir="regions", sequence_length=50, n_regions=6, use_all_combinations=False):
    """
    Load alcohol concentration patch data and create samples by batching regions.
    
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
    
    Returns:
    --------
    data : dict
        Contains 'patches', 'labels', 'concentrations', 'concentration_names', 'label_to_concentration'
    """
    patches_dir = Path(patches_dir)
    
    # Alcohol concentrations to load
    alcohol_concs = [0, 20, 40, 60, 80, 100]
    
    # Group by concentration
    concentration_data = defaultdict(list)
    original_lengths = []
    
    # Load 0% (DI water)
    zero_file = patches_dir / 'session_conc_0_regions.npz'
    if zero_file.exists():
        region_patches = load_region_data(zero_file)
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
            concentration_data[0].append(patch)
    
    # Load other concentrations
    for conc in alcohol_concs[1:]:
        f = patches_dir / f'session_conc_alcohol_{conc}_regions.npz'
        if f.exists():
            region_patches = load_region_data(f)
            for patch in region_patches:
                original_lengths.append(len(patch))
                # Normalize sequence length
                if len(patch) < sequence_length:
                    pad_length = sequence_length - len(patch)
                    patch = np.concatenate([patch, np.repeat(patch[-1:], pad_length, axis=0)], axis=0)
                else:
                    patch = patch[:sequence_length]
                concentration_data[conc].append(patch)
    
    print(f"Loaded data for concentrations: {sorted(concentration_data.keys())}")
    print(f"Original sequence lengths: min={min(original_lengths)}, max={max(original_lengths)}, mean={np.mean(original_lengths):.1f}")
    
    # Determine common spatial dimensions (use the maximum dimensions found)
    all_spatial_dims = []
    for conc, patches in concentration_data.items():
        for patch in patches:
            all_spatial_dims.append(patch.shape[1:])  # (H, W)
    
    max_h = max(dim[0] for dim in all_spatial_dims)
    max_w = max(dim[1] for dim in all_spatial_dims)
    print(f"Standardizing spatial dimensions to: ({max_h}, {max_w})")
    
    # Resize all patches to common dimensions
    for conc in concentration_data.keys():
        resized_patches = []
        for patch in concentration_data[conc]:
            if patch.shape[1] != max_h or patch.shape[2] != max_w:
                # Resize spatial dimensions
                zoom_factors = (1.0, max_h / patch.shape[1], max_w / patch.shape[2])
                resized_patch = zoom(patch, zoom_factors, order=1)
                resized_patches.append(resized_patch)
            else:
                resized_patches.append(patch)
        concentration_data[conc] = resized_patches
    
    # Create labels
    concentration_names = sorted(concentration_data.keys())
    label_to_concentration = {i: conc for i, conc in enumerate(concentration_names)}
    concentration_to_label = {conc: i for i, conc in enumerate(concentration_names)}
    
    # Now create samples by generating all combinations of n_regions
    # Store combinations first, then split, then extract frames to avoid data leakage
    all_combinations = []  # List of (averaged_sequence, label, conc)
    all_combo_labels = []
    all_combo_concentrations = []
    
    for conc, patches in concentration_data.items():
        print(f"\nConcentration {conc}%: {len(patches)} regions")
        
        # Check if we have enough regions
        if len(patches) < n_regions:
            print(f"  Warning: Only {len(patches)} regions available, need at least {n_regions}")
            print(f"  Skipping this concentration")
            continue
        
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
                all_combo_labels.append(concentration_to_label[conc])
                all_combo_concentrations.append(conc)
            
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
                all_combo_labels.append(concentration_to_label[conc])
                all_combo_concentrations.append(conc)
            
            print(f"  Created {num_samples} non-overlapping samples (will expand to {num_samples * sequence_length} frames)")
    
    # Convert combinations to numpy arrays
    all_combinations = np.array(all_combinations)  # (num_combos, seq_len, H, W)
    all_combo_labels = np.array(all_combo_labels)
    all_combo_concentrations = np.array(all_combo_concentrations)
    
    print(f"\nCombination summary:")
    print(f"  Total combinations: {len(all_combinations)}")
    print(f"  Combination shape: {all_combinations.shape}")
    print(f"  Number of classes: {len(concentration_names)}")
    print(f"  Combinations per class:")
    for i, conc in enumerate(concentration_names):
        count = np.sum(all_combo_labels == i)
        print(f"    {conc}%: {count} combinations")
    
    return {
        'combinations': all_combinations,  # (num_combos, seq_len, H, W)
        'combo_labels': all_combo_labels,
        'combo_concentrations': all_combo_concentrations,
        'concentration_names': concentration_names,
        'label_to_concentration': label_to_concentration,
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

def train_test_split_model(data, test_size=0.2, epochs=100, batch_size=16, lr=0.001):
    """Train and evaluate model with train/test split."""
    combinations = data['combinations']  # (num_combos, seq_len, H, W)
    combo_labels = data['combo_labels']
    concentration_names = data['concentration_names']
    sequence_length = data['sequence_length']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Split at the COMBINATION level to avoid data leakage
    print("\n=== Splitting combinations (no data leakage) ===")
    combo_train, combo_test, label_train, label_test = train_test_split(
        combinations, combo_labels, test_size=test_size, random_state=42, stratify=combo_labels
    )
    
    print(f"Train combinations: {len(combo_train)}")
    print(f"Test combinations: {len(combo_test)}")
    
    # Now extract frames from each split
    X_train = []
    y_train = []
    for combo, label in zip(combo_train, label_train):
        for frame_idx in range(combo.shape[0]):
            X_train.append(combo[frame_idx])
            y_train.append(label)
    
    X_test = []
    y_test = []
    for combo, label in zip(combo_test, label_test):
        for frame_idx in range(combo.shape[0]):
            X_test.append(combo[frame_idx])
            y_test.append(label)
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    print(f"\nTrain frames: {len(X_train)} (from {len(combo_train)} combos × {sequence_length} frames)")
    print(f"Test frames: {len(X_test)} (from {len(combo_test)} combos × {sequence_length} frames)")
    
    # Create datasets
    train_dataset = ConcentrationDataset(X_train, y_train, concentration_names)
    test_dataset = ConcentrationDataset(X_test, y_test, concentration_names, 
                                       mean=train_dataset.mean, std=train_dataset.std)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    input_size = X_train.shape[1]  # Height/Width of patch (now 2D frames)
    num_classes = len(concentration_names)
    
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
    
    print(f"\nTraining CNN model (per-frame classification)...")
    
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
        
        if (epoch + 1) % 4 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    
    # Final evaluation with probabilities
    _, final_test_acc, test_preds, test_true, test_probs = evaluate(
        model, test_loader, criterion, device, return_probabilities=True
    )
    
    # Convert to numpy arrays if not already
    test_preds = np.array(test_preds)
    test_true = np.array(test_true)
    test_probs = np.array(test_probs)
    
    print(f"\n{'='*50}")
    print(f"TRAINING RESULTS")
    print(f"{'='*50}")
    print(f"Best test accuracy: {best_test_acc:.4f} at epoch {best_epoch}")
    print(f"Final test accuracy: {final_test_acc:.4f}")
    
    # Debug: print shapes and sample values
    print(f"\nDebug Info:")
    print(f"  test_true shape: {test_true.shape}, unique values: {np.unique(test_true)}")
    print(f"  test_preds shape: {test_preds.shape}, unique values: {np.unique(test_preds)}")
    print(f"  test_probs shape: {test_probs.shape}")
    print(f"  test_probs sample (first 3): {test_probs[:3]}")
    
    # Print average confidence scores per class
    print(f"\n{'='*50}")
    print(f"AVERAGE CONFIDENCE SCORES PER CLASS")
    print(f"{'='*50}")
    for class_idx, conc_name in enumerate(concentration_names):
        # Get samples that truly belong to this class
        class_mask = (test_true == class_idx)
        num_samples = np.sum(class_mask)
        print(f"\nClass {class_idx} ({conc_name}%): {num_samples} samples")
        if num_samples > 0:
            class_probs = test_probs[class_mask]
            avg_confidence = np.mean(class_probs[:, class_idx])
            print(f"  Average confidence: {avg_confidence:.4f}")
            print(f"  Sample probs: {class_probs[0] if len(class_probs) > 0 else 'N/A'}")
    
    return {
        'model': model,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_accs': train_accs,
        'test_accs': test_accs,
        'best_test_acc': best_test_acc,
        'final_test_acc': final_test_acc,
        'test_preds': test_preds,
        'test_true': test_true,
        'test_probs': test_probs,
        'concentration_names': concentration_names
    }

def plot_results(results, save_dir='results', run_name=None):
    """Plot training curves, confusion matrix, and confidence scores."""
    os.makedirs(save_dir, exist_ok=True)
    
    if run_name is None:
        run_name = f"alcohol_conc_perframe"
    
    # Plot training curves and confusion matrix
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    epochs = range(1, len(results['train_losses']) + 1)
    
    # Loss curves
    ax1.plot(epochs, results['train_losses'], 'b-', label='Training Loss', alpha=0.8)
    ax1.plot(epochs, results['test_losses'], 'r-', label='Test Loss', alpha=0.8)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Test Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curves
    ax2.plot(epochs, results['train_accs'], 'b-', label='Training Accuracy', alpha=0.8)
    ax2.plot(epochs, results['test_accs'], 'r-', label='Test Accuracy', alpha=0.8)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Test Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Confusion matrix
    cm = confusion_matrix(results['test_true'], results['test_preds'])
    concentration_names = [f"{conc}%" for conc in results['concentration_names']]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=concentration_names, yticklabels=concentration_names, ax=ax3)
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('Actual')
    ax3.set_title('Confusion Matrix')
    
    # Summary text
    ax4.axis('off')
    summary_text = f"""
    Alcohol Concentration Classification Results
    
    Model: CNN (per-frame)
    Classes: {len(results['concentration_names'])}
    
    Best test accuracy: {results['best_test_acc']:.4f}
    Final test accuracy: {results['final_test_acc']:.4f}
    
    Per-class performance:
    """
    
    # Add per-class metrics
    class_report = classification_report(results['test_true'], results['test_preds'], 
                                       target_names=concentration_names, output_dict=True)
    
    for i, conc_name in enumerate(concentration_names):
        if conc_name in class_report:
            precision = class_report[conc_name]['precision']
            recall = class_report[conc_name]['recall']
            f1 = class_report[conc_name]['f1-score']
            summary_text += f"\n    {conc_name}: P={precision:.3f} R={recall:.3f} F1={f1:.3f}"
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=10, 
             verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_curves_{run_name}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot confidence scores / probabilities
    plot_confidence_scores(results, save_dir, run_name)
    
    # Plot feature representation (PCA on model features)
    plot_feature_vs_concentration(results, save_dir, run_name)
    
    # Save text summary
    with open(f'{save_dir}/summary_{run_name}.txt', 'w') as f:
        f.write(f"Alcohol Concentration Classification Results\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Model: CNN (per-frame)\n")
        f.write(f"Classes: {len(results['concentration_names'])}\n")
        f.write(f"Best test accuracy: {results['best_test_acc']:.4f}\n")
        f.write(f"Final test accuracy: {results['final_test_acc']:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(results['test_true'], results['test_preds'], 
                                     target_names=concentration_names))
    
    print(f"Results saved to {save_dir}/")

def plot_confidence_scores(results, save_dir='results', run_name=None):
    """Plot confidence scores (probabilities) for each class."""
    test_probs = results['test_probs']
    test_true = results['test_true']
    concentration_names = results['concentration_names']
    
    num_classes = len(concentration_names);
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    class_names = [f"{conc}%" for conc in concentration_names]
    
    # For each class, plot the distribution of predicted probabilities
    for class_idx in range(num_classes):
        ax = axes[class_idx]
        
        # Get samples that truly belong to this class
        class_mask = test_true == class_idx
        class_probs = test_probs[class_mask]
        
        if len(class_probs) > 0:
            # Plot histogram for each class probability
            for pred_class_idx in range(num_classes):
                ax.hist(class_probs[:, pred_class_idx], bins=30, alpha=0.5, 
                       label=f'P({class_names[pred_class_idx]})', 
                       color=plt.cm.tab10(pred_class_idx))
            
            ax.set_xlabel('Probability')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Confidence Distribution for True Class: {class_names[class_idx]}')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
    
    # Hide unused subplots if num_classes < 6
    for idx in range(num_classes, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/confidence_distribution_{run_name}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot average confidence per class as bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    avg_confidences = []
    for class_idx in range(num_classes):
        class_mask = test_true == class_idx
        if np.sum(class_mask) > 0:
            class_probs = test_probs[class_mask]
            avg_confidence = np.mean(class_probs[:, class_idx])
            avg_confidences.append(avg_confidence)
        else:
            avg_confidences.append(0.0)
    
    bars = ax.bar(class_names, avg_confidences, color=plt.cm.viridis(np.linspace(0, 1, num_classes)), alpha=0.7)
    ax.set_xlabel('Concentration Class', fontsize=12)
    ax.set_ylabel('Average Confidence Score', fontsize=12)
    ax.set_title('Average Confidence Score for Correct Predictions Per Class', fontsize=14)
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, conf in zip(bars, avg_confidences):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{conf:.3f}',
               ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/avg_confidence_scores_{run_name}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot confidence matrix (mean probability for each true class -> predicted class)
    fig, ax = plt.subplots(figsize=(10, 8))
    
    confidence_matrix = np.zeros((num_classes, num_classes))
    for true_class in range(num_classes):
        class_mask = test_true == true_class
        if np.sum(class_mask) > 0:
            class_probs = test_probs[class_mask]
            confidence_matrix[true_class, :] = np.mean(class_probs, axis=0)
    
    sns.heatmap(confidence_matrix, annot=True, fmt='.3f', cmap='YlOrRd', 
                xticklabels=class_names, yticklabels=class_names, ax=ax,
                vmin=0, vmax=1, cbar_kws={'label': 'Average Probability'})
    ax.set_xlabel('Predicted Class', fontsize=12)
    ax.set_ylabel('True Class', fontsize=12)
    ax.set_title('Average Confidence Matrix (Probability Distribution)', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/confidence_matrix_{run_name}.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_feature_vs_concentration(results, save_dir='results', run_name=None):
    """
    Extract features from the model's penultimate layer and plot the main feature
    (first principal component) vs. concentration.
    """
    from sklearn.decomposition import PCA
    
    model = results['model']
    test_true = results['test_true']
    test_probs = results['test_probs']
    concentration_names = results['concentration_names']
    
    # We need to extract features from the test set
    # Let's recreate the test dataset and extract features
    device = next(model.parameters()).device
    
    # Get the test data from results (we'll need to pass it through)
    # For now, we'll use the logits before softmax as a proxy
    # Better approach: extract features from fc1 layer
    
    # Extract features using a hook
    features = []
    
    def hook_fn(module, input, output):
        features.append(output.detach().cpu().numpy())
    
    # Register hook on the penultimate layer (fc1)
    hook = model.fc1.register_forward_hook(hook_fn)
    
    # We need to get the original test data
    # Since we don't have it directly, we'll work with what we have
    # For visualization, we'll use the output probabilities as a proxy
    
    # Remove hook for now since we don't have direct access to test data
    hook.remove()
    
    # Alternative: Use the probability vectors as features for PCA
    # This is a valid representation of the model's final layer
    feature_vectors = test_probs  # Shape: (n_samples, n_classes)
    
    # Apply PCA to reduce to 1D
    pca = PCA(n_components=1)
    main_feature = pca.fit_transform(feature_vectors)  # Shape: (n_samples, 1)
    main_feature = main_feature.flatten()
    
    # Group by true class
    class_features = {}
    for class_idx, conc_name in enumerate(concentration_names):
        class_mask = (test_true == class_idx)
        class_features[conc_name] = main_feature[class_mask]
    
    # Compute mean and std for each class
    class_means = []
    class_stds = []
    for conc in concentration_names:
        if conc in class_features and len(class_features[conc]) > 0:
            class_means.append(np.mean(class_features[conc]))
            class_stds.append(np.std(class_features[conc]))
        else:
            class_means.append(0.0)
            class_stds.append(0.0)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Line plot with error bars
    ax1.errorbar(concentration_names, class_means, yerr=class_stds, 
                 fmt='o-', capsize=5, color='darkblue', linewidth=2, markersize=8, alpha=0.8)
    ax1.set_xlabel('Alcohol Concentration (%)', fontsize=12)
    ax1.set_ylabel('Main Feature (PC1 of Output Layer)', fontsize=12)
    ax1.set_title('Principal Component 1 of Model Output vs. Concentration', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(concentration_names)
    ax1.set_xticklabels([f'{c}%' for c in concentration_names])
    
    # Add variance explained
    variance_explained = pca.explained_variance_ratio_[0] * 100
    ax1.text(0.02, 0.98, f'Variance Explained: {variance_explained:.1f}%', 
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: Distribution of features per class (violin plot)
    data_for_violin = [class_features[conc] for conc in concentration_names if conc in class_features and len(class_features[conc]) > 0]
    labels_for_violin = [f'{conc}%' for conc in concentration_names if conc in class_features and len(class_features[conc]) > 0]
    
    if data_for_violin:
        parts = ax2.violinplot(data_for_violin, positions=range(len(data_for_violin)), 
                               showmeans=True, showmedians=True)
        ax2.set_xlabel('Alcohol Concentration', fontsize=12)
        ax2.set_ylabel('Main Feature (PC1 of Output Layer)', fontsize=12)
        ax2.set_title('Distribution of Main Feature per Concentration', fontsize=14)
        ax2.set_xticks(range(len(labels_for_violin)))
        ax2.set_xticklabels(labels_for_violin)
        ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/main_feature_vs_concentration_{run_name}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print feature statistics
    print(f"\n{'='*50}")
    print(f"MAIN FEATURE STATISTICS")
    print(f"{'='*50}")
    print(f"PCA Variance Explained by PC1: {variance_explained:.2f}%")
    print(f"\nMean feature value per concentration:")
    for conc, mean_val, std_val in zip(concentration_names, class_means, class_stds):
        print(f"  {conc}%: {mean_val:.4f} ± {std_val:.4f}")

def main():
    """Main training function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train alcohol concentration classifier (per-frame)')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for training (default: 16)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--n-regions', type=int, default=1,
                       help='Number of regions to group and average as one sample (default: 1)')
    parser.add_argument('--use-all-combinations', action='store_true',
                       help='Use all C(N,n) combinations (may cause data leakage). Default: False (use non-overlapping sampling)')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    print("Loading alcohol concentration data...")
    print(f"Sampling mode: {'All combinations (may have data leakage)' if args.use_all_combinations else 'Non-overlapping (no data leakage)'}")
    data = load_alcohol_concentration_data(n_regions=args.n_regions, use_all_combinations=args.use_all_combinations)
    
    if len(data['concentration_names']) < 2:
        print("Error: Need at least 2 concentration classes for classification")
        return
    
    print(f"\nStarting training with CNN model (per-frame)...")
    results = train_test_split_model(data, epochs=20, batch_size=args.batch_size)
    
    # Create run name
    combo_mode = "allcombos" if args.use_all_combinations else "nonoverlap"
    run_name = f"alcohol_conc_perframe_n{args.n_regions}_{combo_mode}"
    
    print("\nPlotting results...")
    plot_results(results, run_name=run_name)
    
    print(f"\nTraining complete! Results saved to results/")

if __name__ == "__main__":
    main()
