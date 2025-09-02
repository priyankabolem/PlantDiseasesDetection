# Model Status and Training Instructions

## Current Status

The model weights in `weights/pretrained/best_model.h5` are demonstration weights designed to show the system functionality. For production use, the model needs to be trained on a real plant disease dataset.

## Why Current Predictions Are Similar

The current model weights were initialized for demonstration purposes and tend to predict similar classes for different images. This is because:

1. The model has not been trained on actual plant disease data
2. The weights are synthetic and don't capture real disease patterns
3. The softmax output layer causes one or few classes to dominate

## How to Train a Production Model

### 1. Obtain Training Data

Download a plant disease dataset such as:
- PlantVillage dataset
- Plant Pathology Challenge dataset
- Custom collected and labeled plant disease images

### 2. Organize Data

Structure the data as:
```
dataset/
├── Apple___Apple_scab/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Apple___Black_rot/
│   └── ...
├── Apple___healthy/
│   └── ...
└── ... (other disease classes)
```

### 3. Train the Model

```bash
python train_model.py \
    --data-dir /path/to/dataset \
    --output-dir training_output \
    --epochs 50 \
    --batch-size 32
```

### 4. Deploy Trained Weights

After training completes:
1. Copy `training_output/best_model.h5` to `weights/pretrained/best_model.h5`
2. Copy `training_output/class_names.json` to `weights/pretrained/class_names.json`
3. Push to repository

## Expected Results After Training

With properly trained weights:
- Apple disease images → Apple disease predictions
- Tomato disease images → Tomato disease predictions  
- Healthy plant images → Healthy predictions
- Confidence scores typically 70-95% for clear images

## Temporary Solution

For demonstration purposes, the model currently functions but may show similar predictions for different images. This is expected behavior until proper training is completed.

## Resources

- [PlantVillage Dataset](https://github.com/spMohanty/PlantVillage-Dataset)
- [TensorFlow Data Augmentation](https://www.tensorflow.org/tutorials/images/data_augmentation)
- [Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)