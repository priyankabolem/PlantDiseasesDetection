# Model Weights Directory

This directory contains trained model weights for the Plant Disease Detection system.

## Directory Structure

```
weights/
├── pretrained/      # Pre-trained models ready for inference
│   ├── best_model.h5
│   ├── best_model.onnx
│   ├── best_model.tflite
│   └── class_names.json
└── checkpoints/     # Training checkpoints
```

## Downloading Pre-trained Weights

To download the pre-trained weights, run:

```bash
python scripts/download_weights.py
```

## Model Formats

- **H5 Format**: TensorFlow/Keras native format, best for Python inference
- **ONNX Format**: Cross-platform format, good for deployment
- **TFLite Format**: Optimized for mobile and edge devices

## Model Information

- **Architecture**: EfficientNet-B0 (default)
- **Input Size**: 224x224x3
- **Number of Classes**: 38
- **Training Dataset**: New Plant Diseases Dataset (87K+ images)