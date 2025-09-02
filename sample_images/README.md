# Sample Images

This directory contains sample images for testing the Plant Disease Detection system.

## Directory Structure

```
samples/
├── healthy/          # Healthy plant leaves
├── diseased/         # Diseased plant leaves
└── test_batch.txt    # List of images for batch testing
```

## Sample Images Included

### Healthy Plants
- apple_healthy.jpg
- tomato_healthy.jpg
- potato_healthy.jpg
- grape_healthy.jpg

### Diseased Plants
- apple_scab.jpg
- tomato_early_blight.jpg
- potato_late_blight.jpg
- grape_black_rot.jpg

## Usage

### Single Image Inference
```bash
python infer.py --model weights/pretrained/best_model.h5 --image samples/diseased/tomato_early_blight.jpg
```

### Batch Inference
```bash
python infer.py --model weights/pretrained/best_model.h5 --batch samples/test_batch.txt
```

### Streamlit Demo
The sample images can be uploaded directly through the Streamlit interface.

## Adding Your Own Samples

1. Add images to appropriate subdirectories
2. Ensure images are in JPG or PNG format
3. For best results, use clear images of individual leaves

## Image Requirements

- **Format**: JPG, PNG
- **Recommended Size**: At least 224x224 pixels
- **Content**: Clear view of plant leaf
- **Background**: Preferably plain background
- **Lighting**: Good, even lighting without harsh shadows