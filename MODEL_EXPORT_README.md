# Model Export for Real-Time Demo

This guide explains how to train and export models for real-time alcohol concentration detection on tablet devices.

## Training and Exporting Models

### Basic Usage

Train a model for alcohol concentration classification and export it for deployment:

```bash
python3 train_coke_spiking_classifier.py \
  --classes unadulterated,ethanol10,ethanol30,ethanol50,ethanol80,ethanol100 \
  --epochs 10 \
  --day-suffix 0109 \
  --sequence-length 50 \
  --save-model
```

This will:
1. Train a CNN model using 5-fold cross-validation
2. Track the best model across all folds (highest accuracy)
3. Export three files:
   - `models/*.pth` - PyTorch model file (for Python deployment)
   - `models/*.onnx` - ONNX model file (for cross-platform deployment)
   - `models/*_metadata.json` - Model metadata including normalization parameters

### Custom Model Directory

Save models to a custom directory:

```bash
python3 train_coke_spiking_classifier.py \
  --classes unadulterated,ethanol10,ethanol30,ethanol50,ethanol80,ethanol100 \
  --epochs 10 \
  --day-suffix 0109 \
  --sequence-length 50 \
  --save-model \
  --model-dir deployed_models
```

## Exported Files

### 1. PyTorch Model (`.pth`)
Contains the complete model state and training information:
- Model architecture weights
- Input/output dimensions
- Class names
- Best accuracy and fold information
- Normalization parameters (mean, std)

### 2. ONNX Model (`.onnx`)
Standard format for cross-platform deployment:
- Optimized for inference
- Compatible with ONNX Runtime (C++, Java, JavaScript, etc.)
- Supports dynamic batch sizes
- Can run on CPU or GPU

### 3. Metadata (`.json`)
Human-readable model information:
```json
{
  "model_type": "SpatioTemporalCNN",
  "input_size": 7,
  "num_classes": 6,
  "spiking_names": ["unadulterated", "ethanol10", "ethanol30", "ethanol50", "ethanol80", "ethanol100"],
  "sequence_length": 50,
  "best_accuracy": 0.9234,
  "normalization": {
    "mean": 0.5234,
    "std": 0.1234
  },
  "input_shape": [7, 7],
  "output_shape": [6]
}
```

## Using the Exported Model

### Python (PyTorch)

```python
import torch
import numpy as np

# Load model
checkpoint = torch.load('models/coke_spiking_model.pth')
model = SpatioTemporalCNN(
    input_size=checkpoint['input_size'],
    num_classes=checkpoint['num_classes']
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Prepare input (single frame: H x W)
frame = np.array([[...]])  # Your sensor data
frame_normalized = (frame - checkpoint['normalization_mean']) / checkpoint['normalization_std']
input_tensor = torch.FloatTensor(frame_normalized).unsqueeze(0)

# Run inference
with torch.no_grad():
    output = model(input_tensor)
    probs = torch.softmax(output, dim=1)
    predicted_class = torch.argmax(probs, dim=1).item()
    confidence = probs[0, predicted_class].item()

print(f"Predicted: {checkpoint['spiking_names'][predicted_class]}")
print(f"Confidence: {confidence:.2%}")
```

### ONNX Runtime (Cross-Platform)

```python
import onnxruntime as ort
import numpy as np
import json

# Load metadata
with open('models/coke_spiking_model_metadata.json', 'r') as f:
    metadata = json.load(f)

# Load ONNX model
session = ort.InferenceSession('models/coke_spiking_model.onnx')

# Prepare input
frame = np.array([[...]], dtype=np.float32)  # Your sensor data
frame_normalized = (frame - metadata['normalization']['mean']) / metadata['normalization']['std']
input_data = frame_normalized.reshape(1, metadata['input_size'], metadata['input_size'])

# Run inference
outputs = session.run(None, {'input': input_data})
logits = outputs[0]

# Get prediction
probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)  # Softmax
predicted_class = np.argmax(probs, axis=1)[0]
confidence = probs[0, predicted_class]

print(f"Predicted: {metadata['spiking_names'][predicted_class]}")
print(f"Confidence: {confidence:.2%}")
```

### Android/iOS (ONNX Runtime Mobile)

The ONNX model can be deployed to mobile devices using ONNX Runtime Mobile:

1. **Android**: Use ONNX Runtime for Android
   - Add dependency: `implementation 'com.microsoft.onnxruntime:onnxruntime-android:latest'`
   
2. **iOS**: Use ONNX Runtime for iOS
   - Add via CocoaPods: `pod 'onnxruntime-mobile'`

Refer to the ONNX Runtime Mobile documentation for platform-specific integration.

## Model Architecture

**SpatioTemporalCNN** (Per-Frame Classifier):
- Input: Single frame (H × W) from averaged regions
- Conv2D layers for spatial feature extraction
- MaxPooling for dimensionality reduction
- Fully connected layers for classification
- Output: Class probabilities (softmax)

## Training Tips

1. **More epochs for better accuracy**:
   ```bash
   --epochs 50
   ```

2. **Use validation data from specific day**:
   ```bash
   --day-suffix 0109
   ```

3. **Adjust sequence length** (frames per sample):
   ```bash
   --sequence-length 50
   ```

4. **Binary classification** (unadulterated vs adulterated):
   ```bash
   --binary --classes unadulterated,ethanol10,ethanol30,ethanol50
   ```

## Deployment Checklist

- [ ] Train model with `--save-model` flag
- [ ] Verify exported files exist in `models/` directory
- [ ] Test PyTorch model loading
- [ ] Test ONNX model inference
- [ ] Implement normalization (mean/std from metadata)
- [ ] Handle real-time frame input
- [ ] Display predicted class and confidence
- [ ] Test on target device (tablet)

## Troubleshooting

**ONNX export fails**: Ensure PyTorch and ONNX are installed:
```bash
pip install torch onnx onnxruntime
```

**Model file not found**: Check that `--save-model` flag is enabled

**Input shape mismatch**: Verify your sensor data matches the expected input size (check metadata)

**Low accuracy**: Try training with more epochs (`--epochs 50`) or use more data
