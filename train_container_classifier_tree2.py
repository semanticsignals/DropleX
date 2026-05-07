#!/usr/bin/env python3
"""
Train a Random Forest classifier for container liquid classification.

This script extracts spatial features from sensor data to capture:
- Ring patterns (positive values around center)
- Center region characteristics (negative values)
- Radial distribution patterns
- Statistical features across different regions

# Train on plcup data (default)
python3 train_container_classifier_tree2.py --liquids tap,di,ethanol100

# Train on heart data
python3 train_container_classifier_tree2.py --container heart --liquids tap,di,ethanol100

# Train on heart data with specific settings
python3 train_container_classifier_tree2.py --container heart --liquids tap,di,ethanol100 --n-estimators 200 --sequence-length 50
"""

import sys
import os
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from collections import defaultdict
from scipy.ndimage import zoom
from itertools import combinations
from scipy import ndimage
import joblib  # For model saving

# Check for help flags early
if '--help' in sys.argv or '-h' in sys.argv:
    print("""usage: train_container_classifier_tree.py [-h] [--n-regions N_REGIONS]
                                          [--liquids LIQUIDS]
                                          [--container CONTAINER]

Train container liquid classifier with Random Forest

optional arguments:
  -h, --help            show this help message and exit
  --seed SEED           Random seed for reproducibility (default: 42)
  --n-regions N_REGIONS
                        Number of regions to group and average as one sample (default: 1)
  --liquids LIQUIDS     Comma-separated list of liquid types to use.
                        Available: tap, di, wine, coke, ckalc10, ethanol100
                        (default: all liquids)
  --container CONTAINER
                        Container type to use: 'plcup' or 'heart' (default: plcup)
  --use-all-combinations
                        Use all C(N,n) combinations (may cause data leakage).
                        Default: False (use non-overlapping random sampling)
  --n-estimators N      Number of trees in random forest (default: 200)
""")
    sys.exit(0)

REGIONS_DIR = 'regions'

# All available liquid types
ALL_LIQUIDS = ['tap', 'di', 'wine', 'coke', 'ckalc10', 'ckalc20', 'ethanol100', 'ckalc40', 'ckalc60', 'ckalc80', 'nacl_0-01', 'nacl_0-001', 'nacl_0-1','nacl_0-0001']

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)

def load_region_data(npz_file):
    """
    Load region data and return list of region patches.
    Each region contains frames as (T, H, W).
    """
    data = np.load(npz_file)
    num_regions = int(data['num_regions'])
    region_patches = []
    
    for region_idx in range(num_regions):
        crops_key = f'region_{region_idx}_crops'
        if crops_key in data:
            crops = data[crops_key]  # Shape: (T, H, W)
            region_patches.append(crops)
    
    return region_patches

def get_feature_names():
    """Return descriptive names for all extracted features."""
    names = [
        'region_pos_mean',      # Average of all positive values
        'region_pos_median',   # Median of all positive values
        'region_pos_75pct',    # 75th percentile of all positive values
    ]
    return names

def extract_spatial_features(frame):
    """
    Extract spatial features from a single frame to capture ring patterns.
    
    Features:
    1. Average of all positive values in the region
    # 2. Maximum value in the region (commented out)
    """
    features = []
    
    # 1. Average of all positive values
    positive_vals = frame[frame > 0]
    if len(positive_vals) > 0:
        features.append(np.mean(positive_vals))
    else:
        features.append(0)
    
    # 2. Maximum value in the region (commented out)
    # features.append(np.max(frame))
    
    return np.array(features)

def load_container_liquid_data(patches_dir="regions", sequence_length=50, n_regions=1, 
                               use_all_combinations=False, selected_liquids=None, container_type='plcup'):
    """
    Load container liquid patch data and create samples by batching regions.
    
    Parameters:
    -----------
    container_type : str
        Type of container: 'plcup' or 'heart' (default: 'plcup')
    
    Returns:
    --------
    data : dict
        Contains 'combinations', 'combo_labels', 'combo_liquids', 
        'liquid_names', 'label_to_liquid', 'sequence_length'
    """
    patches_dir = Path(patches_dir)
    
    # Determine which liquids to load
    if selected_liquids is None:
        liquids_to_load = ALL_LIQUIDS.copy()
    else:
        liquids_to_load = selected_liquids
    
    # Group by liquid type
    liquid_data = defaultdict(list)
    original_lengths = []
    
    for liquid in liquids_to_load:
        # Search for region files matching the liquid type and container
        if container_type == 'plcup':
            if liquid == 'tap':
                pattern = "session_container_plcup_12*_regions.npz"
            else:
                pattern = f"session_container_plcup_{liquid}*_regions.npz"
        elif container_type == 'heart':
            pattern = f"session_container_heart_{liquid}*_regions.npz"
        else:
            raise ValueError(f"Unknown container type: {container_type}. Use 'plcup' or 'heart'.")
        
        files = list(patches_dir.glob(pattern))
        
        if not files:
            print(f"Warning: No region files found for liquid '{liquid}' with container '{container_type}'")
            continue
        
        print(f"Found {len(files)} file(s) for liquid '{liquid}' (container: {container_type}):")
        for f in files:
            print(f"  - {f.name}")
            patches = load_region_data(f)
            if patches:
                liquid_data[liquid].extend(patches)
                original_lengths.extend([p.shape[0] for p in patches])
    
    if not liquid_data:
        raise RuntimeError(f"No data loaded from {patches_dir}")
    
    print(f"\nLoaded data for: {sorted(liquid_data.keys())}")
    print(f"Original sequence lengths: min={min(original_lengths)}, max={max(original_lengths)}, mean={np.mean(original_lengths):.1f}")
    
    # Determine common spatial dimensions
    all_spatial_dims = []
    for liquid, patches in liquid_data.items():
        for patch in patches:
            all_spatial_dims.append(patch.shape[1:])
    
    max_h = max(dim[0] for dim in all_spatial_dims)
    max_w = max(dim[1] for dim in all_spatial_dims)
    print(f"Standardizing spatial dimensions to: ({max_h}, {max_w})")
    
    # Resize all patches to common dimensions and sequence length
    for liquid in liquid_data.keys():
        resized_patches = []
        for patch in liquid_data[liquid]:
            T, H, W = patch.shape
            
            # Resize spatial dimensions if needed
            if H != max_h or W != max_w:
                zoom_factors = (1.0, max_h / H, max_w / W)
                patch = zoom(patch, zoom_factors, order=1)
            
            # Crop if too long
            if T > sequence_length:
                patch = patch[:sequence_length]
            # Pad by duplicating last frame if too short
            elif T < sequence_length:
                pad_len = sequence_length - T
                if T > 0:
                    last_frame = patch[-1:]
                    pad_frames = np.repeat(last_frame, pad_len, axis=0)
                    patch = np.concatenate([patch, pad_frames], axis=0)
            
            resized_patches.append(patch)
        
        liquid_data[liquid] = resized_patches
    
    # Create labels
    liquid_names = sorted(liquid_data.keys())
    label_to_liquid = {i: name for i, name in enumerate(liquid_names)}
    liquid_to_label = {name: i for i, name in enumerate(liquid_names)}
    
    # Create samples by generating combinations of n_regions
    all_combinations = []
    all_combo_labels = []
    all_combo_liquids = []
    
    for liquid, patches in liquid_data.items():
        num_patches = len(patches)
        label = liquid_to_label[liquid]
        
        if use_all_combinations:
            if num_patches < n_regions:
                print(f"Warning: Liquid '{liquid}' has only {num_patches} regions, but n_regions={n_regions}. Skipping.")
                continue
            
            region_combinations = list(combinations(range(num_patches), n_regions))
            print(f"Liquid '{liquid}': {num_patches} regions → {len(region_combinations)} combinations")
            
            for combo_indices in region_combinations:
                selected_patches = [patches[i] for i in combo_indices]
                avg_patch = np.mean(selected_patches, axis=0)
                all_combinations.append(avg_patch)
                all_combo_labels.append(label)
                all_combo_liquids.append(liquid)
        else:
            num_samples = num_patches // n_regions
            if num_samples == 0:
                print(f"Warning: Liquid '{liquid}' has only {num_patches} regions, but n_regions={n_regions}. Skipping.")
                continue
            
            indices = np.random.permutation(num_patches)
            print(f"Liquid '{liquid}': {num_patches} regions → {num_samples} non-overlapping samples")
            
            for i in range(num_samples):
                start_idx = i * n_regions
                end_idx = start_idx + n_regions
                group_indices = indices[start_idx:end_idx]
                
                selected_patches = [patches[j] for j in group_indices]
                avg_patch = np.mean(selected_patches, axis=0)
                all_combinations.append(avg_patch)
                all_combo_labels.append(label)
                all_combo_liquids.append(liquid)
    
    all_combinations = np.array(all_combinations)
    all_combo_labels = np.array(all_combo_labels)
    all_combo_liquids = np.array(all_combo_liquids)
    
    print(f"\nCombination summary:")
    print(f"  Total combinations: {len(all_combinations)}")
    print(f"  Combination shape: {all_combinations.shape}")
    print(f"  Number of classes: {len(liquid_names)}")
    print(f"  Combinations per class:")
    for i, name in enumerate(liquid_names):
        count = np.sum(all_combo_labels == i)
        print(f"    {name}: {count}")
    
    return {
        'combinations': all_combinations,
        'combo_labels': all_combo_labels,
        'combo_liquids': all_combo_liquids,
        'liquid_names': liquid_names,
        'label_to_liquid': label_to_liquid,
        'sequence_length': sequence_length
    }

def extract_features_from_data(combinations, combo_labels):
    """
    Extract features from all frames in all combinations.
    
    Returns:
    --------
    X : array of shape (n_samples, n_features)
    y : array of shape (n_samples,)
    """
    print("\nExtracting spatial features from frames...")
    
    X = []
    y = []
    
    for combo_idx, (combo, label) in enumerate(zip(combinations, combo_labels)):
        # combo shape: (T, H, W)
        for frame_idx, frame in enumerate(combo):
            features = extract_spatial_features(frame)
            X.append(features)
            y.append(label)
        
        if (combo_idx + 1) % 10 == 0:
            print(f"  Processed {combo_idx + 1}/{len(combinations)} combinations...")
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\nFeature extraction complete:")
    print(f"  Feature matrix shape: {X.shape}")
    print(f"  Number of features: {X.shape[1]}")
    print(f"  Number of samples: {len(X)}")
    
    return X, y

def extract_region_feature(region):
    """
    Extract region-level features:
    - region_pos_mean: average pos_mean over all frames
    - region_pos_median: median of pos_mean over all frames
    - region_pos_75pct: 75th percentile of pos_mean over all frames
    region: (T, H, W)
    Returns: list of features
    """
    pos_means = []
    for frame in region:
        positive_vals = frame[frame > 0]
        if len(positive_vals) > 0:
            pos_means.append(np.mean(positive_vals))
        else:
            pos_means.append(0)
    pos_means = np.array(pos_means)
    return [
        float(np.mean(pos_means)),
        float(np.median(pos_means)),
        float(np.percentile(pos_means, 75)),
    ]

def train_random_forest_cv(data, n_folds=5, n_estimators=200):
    combinations = data['combinations']
    combo_labels = data['combo_labels']
    combo_liquids = data['combo_liquids']
    liquid_names = data['liquid_names']

    print(f"\n=== {n_folds}-Fold Cross-Validation (Random Forest, PER-FRAME) ===")

    # PER-FRAME FEATURE EXTRACTION
    print(f"\n{'='*80}")
    print(f"PER-FRAME FEATURE VALUES BY LIQUID TYPE")
    print(f"{'='*80}")
    feature_names = get_feature_names()
    frame_features = []
    frame_labels = []
    frame_liquids = []
    for liquid_idx, liquid_name in enumerate(liquid_names):
        liquid_mask = combo_labels == liquid_idx
        liquid_combos = combinations[liquid_mask]
        print(f"\n{liquid_name.upper()}:")
        print(f"  Number of regions: {len(liquid_combos)}")
        for region_idx, region in enumerate(liquid_combos):
            for frame_idx, frame in enumerate(region):
                positive_vals = frame[frame > 0]
                if len(positive_vals) > 0:
                    mean_val = float(np.mean(positive_vals))
                    median_val = float(np.median(positive_vals))
                    pct75_val = float(np.percentile(positive_vals, 75))
                else:
                    mean_val = 0.0
                    median_val = 0.0
                    pct75_val = 0.0
                feats = [mean_val, median_val, pct75_val]
                frame_features.append(feats)
                frame_labels.append(liquid_idx)
                frame_liquids.append(liquid_name)
                # print(f"  Region {region_idx + 1}, Frame {frame_idx + 1}: " + ", ".join([f"{name}={val:8.2f}" for name, val in zip(feature_names, feats)]))
    print(f"\n{'='*80}\n")
    X = np.array(frame_features)
    y = np.array(frame_labels)

    # Cross-validation
    num_classes = len(liquid_names)
    class_counts = [np.sum(y == i) for i in range(num_classes)]
    min_class_count = min(class_counts)
    if min_class_count >= n_folds:
        print(f"Using StratifiedKFold (min class count: {min_class_count})")
        kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        splits = kfold.split(X, y)
    else:
        print(f"Using regular KFold (min class count: {min_class_count} < {n_folds})")
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        splits = kfold.split(X)

    fold_results = []
    all_test_preds = []
    all_test_true = []
    all_feature_importances = []
    best_model = None
    best_acc = -1
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        print(f"\n--- Fold {fold_idx + 1}/{n_folds} ---")
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_test = X[test_idx]
        y_test = y[test_idx]
        print(f"Train: {len(X_train)} frames")
        print(f"Test:  {len(X_test)} frames")
        print(f"Number of features: {X_train.shape[1]}")
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        rf.fit(X_train, y_train)
        train_preds = rf.predict(X_train)
        test_preds = rf.predict(X_test)
        train_acc = accuracy_score(y_train, train_preds)
        test_acc = accuracy_score(y_test, test_preds)
        # Track best model
        if test_acc > best_acc:
            best_acc = test_acc
            best_model = rf
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy:  {test_acc:.4f}")
        fold_results.append({
            'fold': fold_idx + 1,
            'train_acc': train_acc,
            'test_acc': test_acc,
        })
        all_test_preds.extend(test_preds)
        all_test_true.extend(y_test)
        all_feature_importances.append(rf.feature_importances_)
    all_test_preds = np.array(all_test_preds)
    all_test_true = np.array(all_test_true)
    avg_train_acc = np.mean([r['train_acc'] for r in fold_results])
    std_train_acc = np.std([r['train_acc'] for r in fold_results])
    avg_test_acc = np.mean([r['test_acc'] for r in fold_results])
    std_test_acc = np.std([r['test_acc'] for r in fold_results])
    print(f"\n{'='*60}")
    print(f"CROSS-VALIDATION RESULTS (PER-FRAME)")
    print(f"{'='*60}")
    print(f"Average train accuracy: {avg_train_acc:.4f} ± {std_train_acc:.4f}")
    print(f"Average test accuracy:  {avg_test_acc:.4f} ± {std_test_acc:.4f}")
    print(f"\nPer-fold results:")
    for result in fold_results:
        print(f"  Fold {result['fold']}: Train={result['train_acc']:.4f}, Test={result['test_acc']:.4f}")
    overall_acc = accuracy_score(all_test_true, all_test_preds)
    print(f"\nOverall accuracy (all folds combined): {overall_acc:.4f}")
    avg_feature_importances = np.mean(all_feature_importances, axis=0)
    print(f"\n{'='*60}")
    print(f"FEATURE IMPORTANCE ANALYSIS (PER-FRAME)")
    print(f"{'='*60}")
    print(f"  {feature_names[0]:25s} - {avg_feature_importances[0]:.4f}")
    
    # Save best model
    os.makedirs('results/container_liquid', exist_ok=True)
    best_model_path = 'results/container_liquid/rf_best_cv_model.joblib'
    joblib.dump(best_model, best_model_path)
    print(f"Best CV model saved to {best_model_path} (test accuracy: {best_acc:.4f})")
    
    # Save best model as ONNX
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]
        onnx_model = convert_sklearn(best_model, initial_types=initial_type, target_opset=9)
        onnx_path = 'results/container_liquid/rf_best_cv_model.onnx'
        with open(onnx_path, 'wb') as f:
            f.write(onnx_model.SerializeToString())
        print(f"Best CV model exported to ONNX (IR version 9): {onnx_path}")
    except ImportError:
        print("skl2onnx not installed. Please install with 'pip install skl2onnx' to export ONNX.")
    except Exception as e:
        print(f"ONNX export failed: {e}")
    
    return {
        'fold_results': fold_results,
        'avg_train_acc': avg_train_acc,
        'std_train_acc': std_train_acc,
        'avg_test_acc': avg_test_acc,
        'std_test_acc': std_test_acc,
        'overall_acc': overall_acc,
        'all_test_preds': all_test_preds,
        'all_test_true': all_test_true,
        'liquid_names': liquid_names,
        'n_folds': n_folds,
        'feature_importances': avg_feature_importances
    }

def plot_results(results, save_dir='results/container_liquid', run_name=None):
    """Plot confusion matrix and feature importances."""
    os.makedirs(save_dir, exist_ok=True)
    
    if run_name is None:
        run_name = "rf_classification"
    
    liquid_names = results['liquid_names']
    feature_names = get_feature_names()
    
    fig = plt.figure(figsize=(20, 6))
    
    # Plot 1: Confusion matrix
    ax1 = plt.subplot(1, 4, 1)
    cm = confusion_matrix(results['all_test_true'], results['all_test_preds'])
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=liquid_names, yticklabels=liquid_names, ax=ax1, vmin=0, vmax=100, cbar=True)
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    ax1.set_title('Confusion Matrix - Percentage')

    # Map liquid names for display
    display_name_map = {
        "coke": "0%",
        "ckalc20": "20%",
        "tap": "Tap water",
        "di": "DI water",
        "ethanol100": "Ethanol",
        "nacl_0-01": "0.01% NaCl",
        "nacl_0-001": "0.001% NaCl"
    }
    display_liquid_names = [display_name_map.get(name, name) for name in liquid_names]

    # Save confusion matrix without color bar, with increased font size
    fig_cm, ax_cm = plt.subplots(figsize=(4.6, 5))
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=display_liquid_names, yticklabels=display_liquid_names, ax=ax_cm, vmin=0, vmax=100, cbar=False,
                annot_kws={"size": 18})
    ax_cm.set_xlabel('Predicted', fontsize=18)
    ax_cm.set_ylabel('Actual', fontsize=18)
    # Title: first line "Overall accuracy:", second line "{xx%} (n=xx)"
    overall_acc_pct = results['overall_acc'] * 100
    n_samples = len(results['all_test_true'])
    ax_cm.set_title(f"Overall accuracy:\n{overall_acc_pct:.1f}% (n={n_samples // 50})", fontsize=20)
    ax_cm.tick_params(axis='x', labelsize=16)
    ax_cm.tick_params(axis='y', labelsize=16)
    # Build filename with class names
    class_str = '_'.join(liquid_names)
    cm_filename = f'{save_dir}/confusion_matrix_{class_str}_{run_name}.png'
    fig_cm.tight_layout()
    fig_cm.savefig(cm_filename, dpi=300, bbox_inches='tight')
    fig_cm.show()
    plt.close(fig_cm)
    print(f"Confusion matrix saved to {cm_filename}")
    
    # Plot 2: Per-fold accuracy
    ax2 = plt.subplot(1, 4, 2)
    fold_numbers = [r['fold'] for r in results['fold_results']]
    train_accs = [r['train_acc'] for r in results['fold_results']]
    test_accs = [r['test_acc'] for r in results['fold_results']]
    
    x = np.arange(len(fold_numbers))
    width = 0.35
    ax2.bar(x - width/2, train_accs, width, label='Train Accuracy', alpha=0.8)
    ax2.bar(x + width/2, test_accs, width, label='Test Accuracy', alpha=0.8)
    ax2.axhline(y=results['avg_test_acc'], color='r', linestyle='--', alpha=0.5, label='Avg Test')
    ax2.set_xlabel('Fold')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Per-Fold Accuracy')
    ax2.set_xticks(x)
    ax2.set_xticklabels(fold_numbers)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Top 15 feature importances with names
    ax3 = plt.subplot(1, 4, 3)
    importances = results['feature_importances']
    top_n = 15
    top_indices = np.argsort(importances)[-top_n:][::-1]
    top_importances = importances[top_indices]
    
    top_feature_names = [feature_names[i] for i in top_indices]
    y_pos = np.arange(len(top_feature_names))
    
    ax3.barh(y_pos, top_importances, alpha=0.8)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(top_feature_names, fontsize=7)
    ax3.set_xlabel('Importance')
    ax3.set_title(f'Top {top_n} Features')
    ax3.invert_yaxis()
    ax3.grid(True, alpha=0.3, axis='x')
    
    # Plot 4: All feature importances (sorted)
    ax4 = plt.subplot(1, 4, 4)
    sorted_indices = np.argsort(importances)[::-1]
    sorted_importances = importances[sorted_indices]
    
    ax4.bar(range(len(sorted_importances)), sorted_importances, alpha=0.8)
    ax4.set_xlabel('Feature Rank')
    ax4.set_ylabel('Importance')
    ax4.set_title('All Features (Sorted by Importance)')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.axhline(y=np.mean(importances), color='r', linestyle='--', 
                alpha=0.5, label=f'Mean: {np.mean(importances):.4f}')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/rf_results_{run_name}.png', dpi=300, bbox_inches='tight')
    # plt.show()
    
    # Save detailed feature importance analysis
    with open(f'{save_dir}/rf_feature_importance_{run_name}.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("FEATURE IMPORTANCE ANALYSIS\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Total features: {len(importances)}\n")
        f.write(f"Mean importance: {np.mean(importances):.6f}\n")
        f.write(f"Std importance: {np.std(importances):.6f}\n\n")
        
        f.write("All Features (Ranked by Importance):\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Rank':<6} {'Feature Name':<30} {'Importance':<12} {'Cumulative'}\n")
        f.write("-"*80 + "\n")
        
        cumulative = 0
        for rank, idx in enumerate(sorted_indices, 1):
            cumulative += importances[idx]
            f.write(f"{rank:<6} {feature_names[idx]:<30} {importances[idx]:<12.6f} {cumulative:.4f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("FEATURE GROUPS ANALYSIS\n")
        f.write("="*80 + "\n\n")
        
        # Feature details (only 2 features now)
        f.write("Feature Details:\n")
        f.write("-"*80 + "\n")
        for idx, name in enumerate(feature_names):
            f.write(f"  {name:<30} {importances[idx]:.6f}\n")
        f.write("\n")
    
    # Save text summary
    summary_text = f"""
Random Forest Container Liquid Classification Results
{results['n_folds']}-Fold Cross-Validation

Classes: {len(liquid_names)}
Liquids: {', '.join(liquid_names)}

Average Train Accuracy: {results['avg_train_acc']:.4f} ± {results['std_train_acc']:.4f}
Average Test Accuracy:  {results['avg_test_acc']:.4f} ± {results['std_test_acc']:.4f}
Overall Accuracy: {results['overall_acc']:.4f}

Per-fold Results:
"""
    
    for result in results['fold_results']:
        summary_text += f"  Fold {result['fold']}: Train={result['train_acc']:.4f}, Test={result['test_acc']:.4f}\n"
    
    summary_text += f"\nPer-class performance:\n"
    class_report = classification_report(results['all_test_true'], results['all_test_preds'],
                                       target_names=liquid_names, output_dict=True)
    
    for liquid_name in liquid_names:
        metrics = class_report[liquid_name]
        summary_text += f"  {liquid_name}: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}\n"
    
    with open(f'{save_dir}/rf_summary_{run_name}.txt', 'w') as f:
        f.write(summary_text)
        f.write(f"\n\nFull Classification Report:\n")
        f.write(classification_report(results['all_test_true'], results['all_test_preds'],
                                     target_names=liquid_names))
    
    print(f"\nResults saved to {save_dir}/")
    print(f"  - Plots: rf_results_{run_name}.png")
    print(f"  - Summary: rf_summary_{run_name}.txt")
    print(f"  - Feature Analysis: rf_feature_importance_{run_name}.txt")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Train Random Forest for container liquid classification')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--n-regions', type=int, default=1,
                       help='Number of regions to group and average (default: 1)')
    parser.add_argument('--liquids', type=str, default=None,
                       help='Comma-separated list of liquid types (default: all)')
    parser.add_argument('--container', type=str, default='plcup', choices=['plcup', 'heart'],
                       help='Container type: plcup or heart (default: plcup)')
    parser.add_argument('--use-all-combinations', action='store_true',
                       help='Use all C(N,n) combinations (may cause data leakage)')
    parser.add_argument('--n-estimators', type=int, default=200,
                       help='Number of trees in random forest (default: 200)')
    parser.add_argument('--group1', type=str, default=None,
                       help='Comma-separated list of liquid types for binary group 1')
    parser.add_argument('--group2', type=str, default=None,
                       help='Comma-separated list of liquid types for binary group 2')
    parser.add_argument('--sequence-length', type=str, default='50',
                       help='Number of frames to extract from each sample (default: 50). Can provide comma-separated list (e.g., "50,20,10,5").')
    args = parser.parse_args()
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
    
    # Parse selected liquids
    if args.liquids is not None:
        selected_liquids = [l.strip() for l in args.liquids.split(',')]
        invalid = [l for l in selected_liquids if l not in ALL_LIQUIDS]
        if invalid:
            print(f"Error: Invalid liquid types: {invalid}")
            print(f"Available: {ALL_LIQUIDS}")
            return
    else:
        selected_liquids = None
    # Parse binary groups
    binary_mode = False
    group1_liquids = None
    group2_liquids = None
    if args.group1 is not None and args.group2 is not None:
        group1_liquids = [l.strip() for l in args.group1.split(',') if l.strip()]
        group2_liquids = [l.strip() for l in args.group2.split(',') if l.strip()]
        all_groups = set(group1_liquids + group2_liquids)
        invalid = [l for l in all_groups if l not in ALL_LIQUIDS]
        if invalid:
            print(f"Error: Invalid liquid types in groups: {invalid}")
            print(f"Available: {ALL_LIQUIDS}")
            return
        binary_mode = True
        selected_liquids = group1_liquids + group2_liquids
    
    # Store all results if running multiple sequence lengths
    all_seq_len_results = []
    
    # Run training for each sequence length
    for seq_idx, sequence_length in enumerate(sequence_lengths):
        if multiple_sequence_lengths:
            print(f"\n{'='*70}")
            print(f"EXPERIMENT {seq_idx + 1}/{len(sequence_lengths)}: Sequence Length = {sequence_length}")
            print(f"{'='*70}\n")
    
        print("Loading container liquid data...")
        print(f"Container type: {args.container}")
        print(f"Sequence length: {sequence_length} frames per sample")
        print(f"Sampling mode: {'All combinations (may have data leakage)' if args.use_all_combinations else 'Non-overlapping (no data leakage)'}")
        data = load_container_liquid_data(
            n_regions=args.n_regions,
            sequence_length=sequence_length,
            use_all_combinations=args.use_all_combinations,
            selected_liquids=selected_liquids,
            container_type=args.container
        )
        if len(data['liquid_names']) < 2:
            print("Error: Need at least 2 liquid types for classification")
            return
        # Remap labels for binary classification if needed
        if binary_mode:
            print(f"\nBinary classification mode: group1={group1_liquids}, group2={group2_liquids}")
            # Map each label to 0 (group1) or 1 (group2)
            label_map = {}
            for i, name in enumerate(data['liquid_names']):
                if name in group1_liquids:
                    label_map[i] = 0
                elif name in group2_liquids:
                    label_map[i] = 1
            # Remap combo_labels
            data['combo_labels'] = np.array([label_map[l] for l in data['combo_labels']])
            # Update liquid_names for reporting
            data['liquid_names'] = ['group1', 'group2']
        
        print(f"\nStarting 5-fold cross-validation with Random Forest ({args.n_estimators} trees)...")
        results = train_random_forest_cv(data, n_folds=5, n_estimators=args.n_estimators)
        
        combo_mode = "allcombos" if args.use_all_combinations else "nonoverlap"
        seq_len_str = f"_seq{sequence_length}" if sequence_length != 50 else ""
        container_str = f"_{args.container}" if args.container != 'plcup' else ""
        run_name = f"rf_n{args.n_regions}_{combo_mode}_trees{args.n_estimators}{seq_len_str}{container_str}"
        if binary_mode:
            run_name += "_binary"
        
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
            'task': 'container_liquid',
            'sequence_lengths': [r[0] for r in all_seq_len_results],
            'overall_accuracies': [r[1]['overall_acc'] for r in all_seq_len_results]
        }
        
        import json
        summary_file = 'results/sequence_length_comparison/container_liquid.json'
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        print(f"Sequence length comparison data saved to {summary_file}")
        print(f"\nAll experiments complete! Results saved to results/")
    else:
        print(f"\nTraining complete! Results saved to results/container_liquid/")

    # # === Save final trained model on all data ===
    # print("\nTraining final model on all data and saving...")
    # X = []
    # y = []
    # for combo, label in zip(data['combinations'], data['combo_labels']):
    #     for frame in combo:
    #         features = extract_spatial_features(frame)
    #         X.append(features)
    #         y.append(label)
    # X = np.array(X)
    # y = np.array(y)
    # rf_final = RandomForestClassifier(
    #     n_estimators=args.n_estimators,
    #     max_depth=None,
    #     min_samples_split=5,
    #     min_samples_leaf=2,
    #     max_features='sqrt',
    #     random_state=args.seed,
    #     n_jobs=-1,
    #     verbose=0
    # )
    # rf_final.fit(X, y)
    # os.makedirs('results/container_liquid', exist_ok=True)
    # model_path = f'results/container_liquid/rf_final_model.joblib'
    # joblib.dump(rf_final, model_path)
    # print(f"Final trained model saved to {model_path}")

if __name__ == "__main__":
    main()
