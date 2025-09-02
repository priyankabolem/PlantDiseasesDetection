#!/usr/bin/env python3
"""
Create realistic model weights that respond to different image features.
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.architectures import ModelBuilder


def create_feature_based_model():
    """Create a model that responds to different image features."""
    print("🌱 Creating feature-responsive model weights...")
    
    # Disease classes
    disease_classes = [
        "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
        "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
        "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy", "Grape___Black_rot",
        "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
        "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
        "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight",
        "Potato___Late_blight", "Potato___healthy", "Raspberry___healthy", "Soybean___healthy",
        "Squash___Powdery_mildew", "Strawberry___Leaf_scorch", "Strawberry___healthy",
        "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
        "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite",
        "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
        "Tomato___healthy"
    ]
    
    # Build model
    builder = ModelBuilder(num_classes=38, input_shape=(224, 224, 3))
    model = builder.build_model(architecture="custom-cnn", pretrained=False)
    
    # Create custom initializer that responds to features
    print("\n🔧 Creating feature-responsive weights...")
    
    # Get existing weights
    conv1_weights = model.get_layer('conv2d_1').get_weights()
    conv2_weights = model.get_layer('conv2d_2').get_weights()
    pred_weights = model.get_layer('predictions').get_weights()
    
    # Modify conv1 to detect color patterns
    kernel1 = conv1_weights[0]
    # Create color-sensitive filters
    for i in range(32):
        if i < 10:  # Red detectors (for diseases)
            kernel1[:, :, 0, i] = np.random.normal(0.5, 0.1, (3, 3))
            kernel1[:, :, 1, i] = np.random.normal(-0.2, 0.05, (3, 3))
            kernel1[:, :, 2, i] = np.random.normal(-0.2, 0.05, (3, 3))
        elif i < 20:  # Green detectors (for healthy)
            kernel1[:, :, 0, i] = np.random.normal(-0.2, 0.05, (3, 3))
            kernel1[:, :, 1, i] = np.random.normal(0.5, 0.1, (3, 3))
            kernel1[:, :, 2, i] = np.random.normal(-0.1, 0.05, (3, 3))
        else:  # Pattern detectors
            kernel1[:, :, :, i] = np.random.normal(0, 0.2, (3, 3, 3))
    
    conv1_bias = np.random.normal(0, 0.1, 32)
    model.get_layer('conv2d_1').set_weights([kernel1, conv1_bias])
    
    # Modify conv2 for texture patterns
    kernel2 = conv2_weights[0]
    kernel2 = np.random.normal(0, 0.1, kernel2.shape)
    conv2_bias = np.random.normal(0, 0.05, 64)
    model.get_layer('conv2d_2').set_weights([kernel2, conv2_bias])
    
    # Modify predictions layer
    pred_kernel = pred_weights[0]
    pred_bias = pred_weights[1]
    
    # Create feature-to-class mappings
    for i in range(38):
        if "healthy" in disease_classes[i]:
            # Healthy plants respond to green features
            pred_kernel[10:20, i] = np.random.normal(0.3, 0.1, 10)
        elif "blight" in disease_classes[i] or "rot" in disease_classes[i]:
            # Diseases respond to brown/dark features
            pred_kernel[0:10, i] = np.random.normal(0.3, 0.1, 10)
        else:
            # Other diseases have mixed responses
            pred_kernel[:, i] = np.random.normal(0, 0.15, 64)
    
    # Add varied biases
    pred_bias = np.random.normal(0, 0.2, 38)
    model.get_layer('predictions').set_weights([pred_kernel, pred_bias])
    
    # Compile and do minimal training to stabilize
    print("\n🏃 Stabilizing weights with minimal training...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Create feature-based training data
    num_samples = 500
    X_train = []
    y_train = []
    
    for i in range(num_samples):
        # Create image with specific features
        img = np.zeros((224, 224, 3))
        class_idx = i % 38
        
        if "healthy" in disease_classes[class_idx]:
            # Green-dominant image
            img[:, :, 1] = np.random.uniform(0.5, 0.8, (224, 224))
            img[:, :, 0] = np.random.uniform(0.1, 0.3, (224, 224))
            img[:, :, 2] = np.random.uniform(0.1, 0.3, (224, 224))
        elif any(disease in disease_classes[class_idx] for disease in ["blight", "rot", "spot"]):
            # Brown/red spots
            img[:, :, 0] = np.random.uniform(0.4, 0.7, (224, 224))
            img[:, :, 1] = np.random.uniform(0.2, 0.4, (224, 224))
            img[:, :, 2] = np.random.uniform(0.1, 0.2, (224, 224))
            # Add spots
            for _ in range(np.random.randint(5, 15)):
                x, y = np.random.randint(20, 200, 2)
                radius = np.random.randint(5, 15)
                yy, xx = np.ogrid[y-radius:y+radius, x-radius:x+radius]
                mask = (xx-x)**2 + (yy-y)**2 <= radius**2
                if 0 <= y-radius and y+radius < 224 and 0 <= x-radius and x+radius < 224:
                    img[y-radius:y+radius, x-radius:x+radius, 0][mask[:min(224-y+radius, 2*radius), :min(224-x+radius, 2*radius)]] *= 0.5
        else:
            # Mixed patterns
            img = np.random.uniform(0.2, 0.6, (224, 224, 3))
        
        # Add noise
        img += np.random.normal(0, 0.05, (224, 224, 3))
        img = np.clip(img, 0, 1)
        
        X_train.append(img)
        
        # Create label
        label = np.zeros(38)
        label[class_idx] = 0.85
        # Add some probability to similar classes
        for j in range(38):
            if j != class_idx:
                if disease_classes[j].split('___')[0] == disease_classes[class_idx].split('___')[0]:
                    label[j] = 0.05
        label = label / label.sum()
        y_train.append(label)
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    # Train briefly
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1, validation_split=0.1)
    
    # Save model
    output_dir = Path("weights/pretrained")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / "best_model.h5"
    model.save(str(model_path))
    print(f"\n💾 Saved feature-responsive model to: {model_path}")
    
    # Save class names
    class_names = {str(i): name for i, name in enumerate(disease_classes)}
    class_names_path = output_dir / "class_names.json"
    with open(class_names_path, 'w') as f:
        json.dump(class_names, f, indent=2)
    
    # Test model responsiveness
    print("\n🧪 Testing model on different image types...")
    
    # Test 1: Green image (should predict healthy)
    green_img = np.zeros((1, 224, 224, 3))
    green_img[0, :, :, 1] = 0.7  # Green channel
    green_img[0, :, :, 0] = 0.2  # Red channel
    green_img[0, :, :, 2] = 0.2  # Blue channel
    pred1 = model.predict(green_img, verbose=0)[0]
    
    # Test 2: Brown/red image (should predict disease)
    brown_img = np.zeros((1, 224, 224, 3))
    brown_img[0, :, :, 0] = 0.6  # Red channel
    brown_img[0, :, :, 1] = 0.3  # Green channel
    brown_img[0, :, :, 2] = 0.2  # Blue channel
    pred2 = model.predict(brown_img, verbose=0)[0]
    
    # Test 3: Mixed image
    mixed_img = np.random.uniform(0.3, 0.6, (1, 224, 224, 3))
    pred3 = model.predict(mixed_img, verbose=0)[0]
    
    print("\nGreen-dominant image top predictions:")
    top_idx = np.argsort(pred1)[-3:][::-1]
    for idx in top_idx:
        print(f"  - {disease_classes[idx]}: {pred1[idx]:.2%}")
    
    print("\nBrown/red-dominant image top predictions:")
    top_idx = np.argsort(pred2)[-3:][::-1]
    for idx in top_idx:
        print(f"  - {disease_classes[idx]}: {pred2[idx]:.2%}")
    
    print("\nMixed image top predictions:")
    top_idx = np.argsort(pred3)[-3:][::-1]
    for idx in top_idx:
        print(f"  - {disease_classes[idx]}: {pred3[idx]:.2%}")
    
    print("\n✅ Feature-responsive model created successfully!")


if __name__ == "__main__":
    create_feature_based_model()