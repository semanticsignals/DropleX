#!/usr/bin/env python3
"""
Train a CNN+LSTM model for alcohol concentration classification.

The model uses:
- CNN layers to extract spatial features from each frame
- LSTM layers to model temporal dynamics
- Train/test split for evaluation
- Region batching: randomly group n regions and average them as a sample
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
                                          [--seed SEED] [--model {lstm,transformer}]
                                          [--temporal {sequential,average}]
                                          [--n-regions N_REGIONS]

Train alcohol concentration classifier with CNN+LSTM

optional arguments:
  -h, --help            show this help message and exit
  --batch-size BATCH_SIZE
                        Batch size for training (default: 16)
  --seed SEED           Random seed for reproducibility (default: 42)
  --model {lstm,transformer}
                        Model architecture: lstm (CNN+LSTM) or transformer
                        (CNN+Transformer) (default: lstm)
  --temporal {sequential,average}
                        Temporal method: sequential (LSTM/Transformer) or
                        average (frame averaging) (default: sequential)
  --n-regions N_REGIONS
                        Number of regions to randomly group and average as
                        one sample (default: 6)
  --info                Show detailed information about training
                        configurations""")
    sys.exit(0)

if '--info' in sys.argv:
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║           ALCOHOL CONCENTRATION CLASSIFIER TRAINING                  ║
╚══════════════════════════════════════════════════════════════════════╝

This script trains a CNN+LSTM model to classify alcohol concentrations from
temporal sensor patches.

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
For each concentration, randomly group n regions (default n=6) and average
them to form a single sample. All combinations of these batches are used
as training/test samples.

──────────────────────────────────────────────────────────────────────
EXAMPLES
──────────────────────────────────────────────────────────────────────
# Train with default settings (n=6 regions per sample)
python3 train_conc_alcohol_classifier.py

# Train with different batch size
python3 train_conc_alcohol_classifier.py --n-regions 4

# Train with transformer model
python3 train_conc_alcohol_classifier.py --model transformer

# Train with frame averaging instead of LSTM
python3 train_conc_alcohol_classifier.py --temporal average

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
    """CNN+LSTM model for spatiotemporal concentration classification."""

    def __init__(self, input_size=7, num_classes=6, hidden_dim=64, lstm_layers=2):
        super(SpatioTemporalCNN, self).__init__()
        
        # CNN layers for spatial feature extraction
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        
        # Calculate CNN output size after two pooling operations
        conv_output_size = 64 * ((input_size // 4) ** 2)
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(conv_output_size, hidden_dim, lstm_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        batch_size, seq_len, H, W = x.size()
        
        # Process each frame through CNN
        c_in = x.view(batch_size * seq_len, 1, H, W)
        c_out = torch.relu(self.conv1(c_in))
        c_out = self.pool(c_out)
        c_out = torch.relu(self.conv2(c_out))
        c_out = self.pool(c_out)
        c_out = self.dropout(c_out)
        
        # Flatten spatial dimensions
        c_out = c_out.view(batch_size, seq_len, -1)
        
        # LSTM
        lstm_out, _ = self.lstm(c_out)
        lstm_out = lstm_out[:, -1, :]  # Take last timestep
        
        # Classification
        out = self.fc(lstm_out)
        return out

class FrameAveragingCNN(nn.Module):
    """CNN model using frame averaging instead of temporal modeling."""

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
        batch_size, seq_len, H, W = x.size()
        
        # Process each frame through CNN and average
        c_in = x.view(batch_size * seq_len, 1, H, W)
        c_out = torch.relu(self.conv1(c_in))
        c_out = self.pool(c_out)
        c_out = torch.relu(self.conv2(c_out))
        c_out = self.pool(c_out)
        c_out = self.dropout(c_out)
        
        # Flatten and average across time
        c_out = c_out.view(batch_size, seq_len, -1)
        c_out = c_out.mean(dim=1)
        
        # Classification
        out = torch.relu(self.fc1(c_out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out

def load_alcohol_concentration_data(patches_dir="regions", sequence_length=50, n_regions=6):
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
    all_patches = []
    all_labels = []
    all_concentrations = []
    
    for conc, patches in concentration_data.items():
        print(f"\nConcentration {conc}%: {len(patches)} regions")
        
        # Check if we have enough regions
        if len(patches) < n_regions:
            print(f"  Warning: Only {len(patches)} regions available, need at least {n_regions}")
            print(f"  Skipping this concentration")
            continue
        
        # Generate all combinations of n_regions from available patches
        all_combos = list(combinations(range(len(patches)), n_regions))
        print(f"  Generating C({len(patches)}, {n_regions}) = {len(all_combos)} combinations")
        
        # Create a sample for each combination
        for combo_indices in all_combos:
            # Get the patches for this combination
            combo_patches = [patches[i] for i in combo_indices]
            
            # Average the patches
            combo_array = np.array(combo_patches)  # (n_regions, seq_len, H, W)
            averaged_sample = np.mean(combo_array, axis=0)  # (seq_len, H, W)
            
            all_patches.append(averaged_sample)
            all_labels.append(concentration_to_label[conc])
            all_concentrations.append(conc)
        
        print(f"  Created {len(all_combos)} samples")
    
    # Convert to numpy arrays
    all_patches = np.array(all_patches)
    all_labels = np.array(all_labels)
    all_concentrations = np.array(all_concentrations)
    
    print(f"\nDataset summary:")
    print(f"  Total samples: {len(all_patches)}")
    print(f"  Patch shape: {all_patches.shape}")
    print(f"  Number of classes: {len(concentration_names)}")
    print(f"  Class distribution:")
    for i, conc in enumerate(concentration_names):
        count = np.sum(all_labels == i)
        print(f"    {conc}%: {count} samples")
    
    return {
        'patches': all_patches,
        'labels': all_labels,
        'concentrations': all_concentrations,
        'concentration_names': concentration_names,
        'label_to_concentration': label_to_concentration
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

def evaluate(model, test_loader, criterion, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for patches, labels in test_loader:
            patches, labels = patches.to(device), labels.to(device)
            outputs = model(patches)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(test_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy, all_preds, all_labels

def train_test_split_model(data, test_size=0.2, epochs=100, batch_size=16, lr=0.001, 
                           model_type='lstm', temporal_method='sequential'):
    """Train and evaluate model with train/test split."""
    patches = data['patches']
    labels = data['labels']
    concentration_names = data['concentration_names']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        patches, labels, test_size=test_size, random_state=42, stratify=labels
    )
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Create datasets
    train_dataset = ConcentrationDataset(X_train, y_train, concentration_names)
    test_dataset = ConcentrationDataset(X_test, y_test, concentration_names, 
                                       mean=train_dataset.mean, std=train_dataset.std)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    input_size = patches.shape[2]  # Height/Width of patch
    num_classes = len(concentration_names)
    
    if temporal_method == 'average':
        model = FrameAveragingCNN(input_size=input_size, num_classes=num_classes)
    else:
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
    
    print(f"\nTraining {model_type.upper()} model with {temporal_method} temporal method...")
    
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
            print(f"Epoch [{epoch+1}/{epochs}] "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    
    # Final evaluation
    _, final_test_acc, test_preds, test_true = evaluate(model, test_loader, criterion, device)
    
    print(f"\n{'='*50}")
    print(f"TRAINING RESULTS")
    print(f"{'='*50}")
    print(f"Best test accuracy: {best_test_acc:.4f} at epoch {best_epoch}")
    print(f"Final test accuracy: {final_test_acc:.4f}")
    
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
        'concentration_names': concentration_names
    }

def plot_results(results, save_dir='results', run_name=None, model_type='lstm'):
    """Plot training curves and confusion matrix."""
    os.makedirs(save_dir, exist_ok=True)
    
    if run_name is None:
        run_name = f"alcohol_conc_{model_type}"
    
    # Plot training curves
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
    
    Model: {model_type.upper()}
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
    
    # Save text summary
    with open(f'{save_dir}/summary_{run_name}.txt', 'w') as f:
        f.write(f"Alcohol Concentration Classification Results\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Model: {model_type.upper()}\n")
        f.write(f"Classes: {len(results['concentration_names'])}\n")
        f.write(f"Best test accuracy: {results['best_test_acc']:.4f}\n")
        f.write(f"Final test accuracy: {results['final_test_acc']:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(results['test_true'], results['test_preds'], 
                                     target_names=concentration_names))
    
    print(f"Results saved to {save_dir}/")

def main():
    """Main training function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train alcohol concentration classifier')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for training (default: 16)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--model', choices=['lstm', 'transformer'], default='lstm',
                       help='Model architecture (default: lstm)')
    parser.add_argument('--temporal', choices=['sequential', 'average'],
                       default='sequential', help='Temporal method (default: sequential)')
    parser.add_argument('--n-regions', type=int, default=6,
                       help='Number of regions to group and average as one sample (default: 6)')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    print("Loading alcohol concentration data...")
    data = load_alcohol_concentration_data(n_regions=args.n_regions)
    
    if len(data['concentration_names']) < 2:
        print("Error: Need at least 2 concentration classes for classification")
        return
    
    print(f"\nStarting training with {args.model} model...")
    results = train_test_split_model(data, epochs=100, batch_size=args.batch_size,
                                     model_type=args.model, temporal_method=args.temporal)
    
    # Create run name
    run_name = f"alcohol_conc_{args.model}_{args.temporal}_n{args.n_regions}"
    
    print("\nPlotting results...")
    plot_results(results, run_name=run_name, model_type=args.model)
    
    print(f"\nTraining complete! Results saved to results/")

if __name__ == "__main__":
    main()
