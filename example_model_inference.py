#!/usr/bin/env python3
"""
Example script demonstrating how to load and use exported models for inference.

This script shows:
1. Loading PyTorch models
2. Loading ONNX models
3. Running inference on sample data
4. Proper preprocessing with normalization
"""
import torch
import numpy as np
import json
import argparse
from pathlib import Path

# Import the model architecture (ensure train_coke_spiking_classifier.py is in the same directory)
try:
    from train_coke_spiking_classifier import SpatioTemporalCNN
except ImportError:
    print("Warning: Could not import SpatioTemporalCNN. PyTorch inference will not work.")
    print("Make sure train_coke_spiking_classifier.py is in the same directory.")


def load_pytorch_model(model_path):
    """Load a PyTorch model from .pth file."""
    print(f"Loading PyTorch model from: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Create model
    model = SpatioTemporalCNN(
        input_size=checkpoint['input_size'],
        num_classes=checkpoint['num_classes']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Extract metadata
    metadata = {
        'input_size': checkpoint['input_size'],
        'num_classes': checkpoint['num_classes'],
        'spiking_names': checkpoint['spiking_names'],
        'sequence_length': checkpoint['sequence_length'],
        'best_accuracy': checkpoint['best_accuracy'],
        'best_fold': checkpoint['best_fold'],
        'best_epoch': checkpoint['best_epoch'],
        'normalization_mean': checkpoint['normalization_mean'],
        'normalization_std': checkpoint['normalization_std']
    }
    
    print(f"Model loaded successfully!")
    print(f"  Classes: {metadata['spiking_names']}")
    print(f"  Input size: {metadata['input_size']}x{metadata['input_size']}")
    print(f"  Best accuracy: {metadata['best_accuracy']:.4f}")
    print(f"  From fold {metadata['best_fold']}, epoch {metadata['best_epoch']}")
    
    return model, metadata


def load_onnx_model(model_path, metadata_path):
    """Load an ONNX model."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("Error: onnxruntime not installed. Install with: pip install onnxruntime")
        return None, None
    
    print(f"Loading ONNX model from: {model_path}")
    
    # Load ONNX session
    session = ort.InferenceSession(model_path)
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"ONNX model loaded successfully!")
    print(f"  Classes: {metadata['spiking_names']}")
    print(f"  Input size: {metadata['input_size']}x{metadata['input_size']}")
    print(f"  Best accuracy: {metadata['best_accuracy']:.4f}")
    
    return session, metadata


def preprocess_frame(frame, mean, std):
    """Preprocess a single frame with normalization."""
    frame = np.array(frame, dtype=np.float32)
    frame_normalized = (frame - mean) / std
    return frame_normalized


def inference_pytorch(model, frame, metadata):
    """Run inference using PyTorch model."""
    # Preprocess
    frame_normalized = preprocess_frame(
        frame,
        metadata['normalization_mean'],
        metadata['normalization_std']
    )
    
    # Convert to tensor
    input_tensor = torch.FloatTensor(frame_normalized).unsqueeze(0)
    
    # Run inference
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0, predicted_class].item()
        all_probs = probs[0].numpy()
    
    return predicted_class, confidence, all_probs


def inference_onnx(session, frame, metadata):
    """Run inference using ONNX model."""
    # Preprocess
    frame_normalized = preprocess_frame(
        frame,
        metadata['normalization']['mean'],
        metadata['normalization']['std']
    )
    
    # Reshape for ONNX (batch_size, H, W)
    input_data = frame_normalized.reshape(1, metadata['input_size'], metadata['input_size'])
    input_data = input_data.astype(np.float32)
    
    # Run inference
    outputs = session.run(None, {'input': input_data})
    logits = outputs[0]
    
    # Apply softmax
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    predicted_class = np.argmax(probs, axis=1)[0]
    confidence = probs[0, predicted_class]
    all_probs = probs[0]
    
    return predicted_class, confidence, all_probs


def generate_sample_frame(input_size):
    """Generate a random sample frame for testing."""
    return np.random.randn(input_size, input_size)


def main():
    parser = argparse.ArgumentParser(description='Run inference with exported model')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model file (.pth or .onnx)')
    parser.add_argument('--metadata', type=str, default=None,
                       help='Path to metadata JSON file (required for ONNX)')
    parser.add_argument('--sample', action='store_true',
                       help='Use random sample data for testing')
    
    args = parser.parse_args()
    
    model_path = Path(args.model)
    
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        return
    
    # Determine model type
    if model_path.suffix == '.pth':
        # PyTorch model
        model, metadata = load_pytorch_model(model_path)
        model_type = 'pytorch'
    elif model_path.suffix == '.onnx':
        # ONNX model
        if args.metadata is None:
            # Try to find metadata file automatically
            metadata_path = model_path.with_name(model_path.stem + '_metadata.json')
            if not metadata_path.exists():
                print(f"Error: Metadata file not found. Please specify with --metadata")
                return
        else:
            metadata_path = Path(args.metadata)
        
        model, metadata = load_onnx_model(model_path, metadata_path)
        if model is None:
            return
        model_type = 'onnx'
    else:
        print(f"Error: Unsupported model format: {model_path.suffix}")
        print("Supported formats: .pth (PyTorch), .onnx (ONNX)")
        return
    
    # Generate or load sample frame
    if args.sample:
        print("\n" + "="*60)
        print("RUNNING INFERENCE ON SAMPLE DATA")
        print("="*60)
        
        input_size = metadata.get('input_size', 7)
        frame = generate_sample_frame(input_size)
        
        print(f"\nGenerated random sample frame: {frame.shape}")
        print(f"Frame statistics: mean={np.mean(frame):.4f}, std={np.std(frame):.4f}")
        
        # Run inference
        if model_type == 'pytorch':
            predicted_class, confidence, all_probs = inference_pytorch(model, frame, metadata)
            spiking_names = metadata['spiking_names']
        else:  # onnx
            predicted_class, confidence, all_probs = inference_onnx(model, frame, metadata)
            spiking_names = metadata['spiking_names']
        
        # Display results
        print(f"\n{'='*60}")
        print("INFERENCE RESULTS")
        print(f"{'='*60}")
        print(f"Predicted class: {spiking_names[predicted_class]}")
        print(f"Confidence: {confidence:.2%}")
        print(f"\nAll class probabilities:")
        for i, (name, prob) in enumerate(zip(spiking_names, all_probs)):
            marker = " <-- PREDICTED" if i == predicted_class else ""
            print(f"  {name:20s}: {prob:6.2%}{marker}")
    else:
        print("\nModel loaded successfully!")
        print("Use --sample flag to run inference on random sample data")
        print("\nTo use this model in your application:")
        print("1. Load the model using the functions in this script")
        print("2. Preprocess your sensor frames with the normalization parameters")
        print("3. Run inference to get class predictions and confidences")
        print("\nSee MODEL_EXPORT_README.md for detailed usage examples")


if __name__ == "__main__":
    main()
