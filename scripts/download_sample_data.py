#!/usr/bin/env python3
"""
Download sample plant disease dataset for testing
"""

import os
import json
import shutil
import requests
from pathlib import Path
from zipfile import ZipFile
import numpy as np
from PIL import Image


def create_synthetic_dataset(output_dir: Path, num_samples_per_class: int = 50):
    """Create a synthetic dataset for testing model training."""
    print("Creating synthetic plant disease dataset...")
    
    # Define plant disease classes (38 classes as per the model)
    disease_classes = [
        "Apple___Apple_scab",
        "Apple___Black_rot",
        "Apple___Cedar_apple_rust",
        "Apple___healthy",
        "Blueberry___healthy",
        "Cherry_(including_sour)___Powdery_mildew",
        "Cherry_(including_sour)___healthy",
        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
        "Corn_(maize)___Common_rust_",
        "Corn_(maize)___Northern_Leaf_Blight",
        "Corn_(maize)___healthy",
        "Grape___Black_rot",
        "Grape___Esca_(Black_Measles)",
        "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
        "Grape___healthy",
        "Orange___Haunglongbing_(Citrus_greening)",
        "Peach___Bacterial_spot",
        "Peach___healthy",
        "Pepper,_bell___Bacterial_spot",
        "Pepper,_bell___healthy",
        "Potato___Early_blight",
        "Potato___Late_blight",
        "Potato___healthy",
        "Raspberry___healthy",
        "Soybean___healthy",
        "Squash___Powdery_mildew",
        "Strawberry___Leaf_scorch",
        "Strawberry___healthy",
        "Tomato___Bacterial_spot",
        "Tomato___Early_blight",
        "Tomato___Late_blight",
        "Tomato___Leaf_Mold",
        "Tomato___Septoria_leaf_spot",
        "Tomato___Spider_mites Two-spotted_spider_mite",
        "Tomato___Target_Spot",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
        "Tomato___Tomato_mosaic_virus",
        "Tomato___healthy"
    ]
    
    # Create directory structure
    dataset_dir = output_dir / "plant_disease_dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating {len(disease_classes)} class directories...")
    
    # Generate synthetic images for each class
    for i, class_name in enumerate(disease_classes):
        class_dir = dataset_dir / class_name
        class_dir.mkdir(exist_ok=True)
        
        # Generate images with slight variations
        for j in range(num_samples_per_class):
            # Create synthetic leaf image with class-specific patterns
            img_array = create_synthetic_leaf_image(i, j)
            
            # Save image
            img = Image.fromarray(img_array)
            img_path = class_dir / f"{class_name}_{j:04d}.jpg"
            img.save(img_path, "JPEG", quality=85)
        
        print(f"  - Created {num_samples_per_class} images for {class_name}")
    
    # Create metadata
    metadata = {
        "dataset_name": "Synthetic Plant Disease Dataset",
        "num_classes": len(disease_classes),
        "samples_per_class": num_samples_per_class,
        "total_samples": len(disease_classes) * num_samples_per_class,
        "classes": disease_classes,
        "image_size": [224, 224, 3],
        "description": "Synthetic dataset for testing plant disease detection model"
    }
    
    metadata_path = dataset_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nDataset created successfully at: {dataset_dir}")
    print(f"   - Total images: {metadata['total_samples']}")
    print(f"   - Classes: {metadata['num_classes']}")
    
    return dataset_dir


def create_synthetic_leaf_image(class_idx: int, sample_idx: int):
    """Create a synthetic leaf image with class-specific features."""
    # Base parameters
    img_size = 224
    
    # Create base leaf shape (elliptical)
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    
    # Background color (slightly varied)
    bg_color = np.array([240, 245, 240]) + np.random.randint(-10, 10, 3)
    img[:] = bg_color.clip(0, 255)
    
    # Leaf parameters based on class
    np.random.seed(class_idx * 1000 + sample_idx)  # Reproducible randomness
    
    # Leaf color (varies by plant type and health)
    plant_type = class_idx // 4  # Approximate plant grouping
    is_healthy = "healthy" in disease_classes[class_idx]
    
    if is_healthy:
        # Healthy green variations
        leaf_color = np.array([50 + plant_type * 5, 150 + plant_type * 3, 50])
    else:
        # Disease colors (browns, yellows, spots)
        disease_type = class_idx % 8
        if disease_type < 3:  # Fungal - brown spots
            leaf_color = np.array([100, 80, 50])
        elif disease_type < 5:  # Viral - yellow patches
            leaf_color = np.array([180, 170, 80])
        else:  # Bacterial - dark spots
            leaf_color = np.array([60, 70, 40])
    
    # Add noise for variation
    leaf_color = (leaf_color + np.random.randint(-20, 20, 3)).clip(0, 255)
    
    # Create leaf shape
    center_x, center_y = img_size // 2, img_size // 2
    leaf_width = 60 + np.random.randint(-10, 10)
    leaf_height = 80 + np.random.randint(-10, 10)
    
    # Draw elliptical leaf
    for y in range(img_size):
        for x in range(img_size):
            # Check if point is inside ellipse
            dx = (x - center_x) / leaf_width
            dy = (y - center_y) / leaf_height
            if dx*dx + dy*dy <= 1:
                # Add texture
                texture = np.random.randint(-10, 10)
                pixel_color = (leaf_color + texture).clip(0, 255)
                img[y, x] = pixel_color
    
    # Add disease-specific features
    if not is_healthy:
        num_spots = 5 + class_idx % 10
        for _ in range(num_spots):
            # Random spot position on leaf
            spot_x = center_x + np.random.randint(-leaf_width//2, leaf_width//2)
            spot_y = center_y + np.random.randint(-leaf_height//2, leaf_height//2)
            spot_radius = np.random.randint(3, 8)
            
            # Draw disease spot
            for y in range(max(0, spot_y - spot_radius), min(img_size, spot_y + spot_radius)):
                for x in range(max(0, spot_x - spot_radius), min(img_size, spot_x + spot_radius)):
                    if (x - spot_x)**2 + (y - spot_y)**2 <= spot_radius**2:
                        # Check if on leaf
                        dx = (x - center_x) / leaf_width
                        dy = (y - center_y) / leaf_height
                        if dx*dx + dy*dy <= 1:
                            spot_color = leaf_color * 0.6  # Darker spots
                            img[y, x] = spot_color.astype(np.uint8)
    
    # Add slight rotation/perspective
    angle = np.random.randint(-15, 15)
    M = cv2_getRotationMatrix2D((center_x, center_y), angle, 1.0)
    img = cv2_warpAffine(img, M, (img_size, img_size), borderValue=bg_color.tolist())
    
    return img


# Simple rotation without cv2 dependency
def cv2_getRotationMatrix2D(center, angle, scale):
    """Simple rotation matrix calculation."""
    angle_rad = np.radians(angle)
    alpha = scale * np.cos(angle_rad)
    beta = scale * np.sin(angle_rad)
    return np.array([
        [alpha, beta, (1 - alpha) * center[0] - beta * center[1]],
        [-beta, alpha, beta * center[0] + (1 - alpha) * center[1]]
    ])


def cv2_warpAffine(img, M, dsize, borderValue):
    """Simple affine transformation."""
    # For simplicity, just return the original image
    # In production, use proper cv2.warpAffine
    return img


# Initialize disease class names
disease_classes = [
    "Apple___Apple_scab",
    "Apple___Black_rot", 
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]


def main():
    """Main function to create dataset."""
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    
    # Create synthetic dataset
    dataset_path = create_synthetic_dataset(output_dir, num_samples_per_class=30)
    
    print("\n📌 To train the model with this dataset, run:")
    print(f"   python train_model.py --data-dir {dataset_path}")


if __name__ == "__main__":
    main()