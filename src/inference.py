"""Inference module for Plant Disease Detection."""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import tensorflow as tf
from PIL import Image

from .visualization.gradcam import GradCAM
from .data.treatments import get_treatment_recommendation

logger = logging.getLogger(__name__)


class PlantDiseaseClassifier:
    """Plant disease classifier for inference."""

    def __init__(self, model_path: str, class_names_path: Optional[str] = None):
        """Initialize classifier.

        Args:
            model_path: Path to trained model
            class_names_path: Path to class names JSON file
        """
        self.model = self._load_model(model_path)
        self.class_names = self._load_class_names(model_path, class_names_path)
        self.input_shape = self.model.input_shape[1:3]

        # Build the model by running a dummy prediction to initialize it
        dummy_input = np.zeros((1, *self.input_shape, 3))
        _ = self.model(dummy_input)

        # Initialize Grad-CAM
        self.gradcam = GradCAM(self.model)

    def _load_model(self, model_path: str) -> tf.keras.Model:
        """Load trained model."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        model = tf.keras.models.load_model(model_path)
        logger.info(f"Loaded model from {model_path}")
        return model

    def _load_class_names(
        self, model_path: str, class_names_path: Optional[str] = None
    ) -> Dict[int, str]:
        """Load class names."""
        if class_names_path and os.path.exists(class_names_path):
            with open(class_names_path, "r") as f:
                class_names = json.load(f)
        else:
            # Try to find class names in the same directory as model
            model_dir = Path(model_path).parent.parent
            class_names_file = model_dir / "class_names.json"

            if class_names_file.exists():
                with open(class_names_file, "r") as f:
                    class_names = json.load(f)
            else:
                # Use default class names
                logger.warning("Class names file not found, using indices")
                class_names = {str(i): f"Class_{i}" for i in range(38)}

        # Ensure keys are integers
        class_names = {int(k): v for k, v in class_names.items()}
        return class_names

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess image for prediction."""
        # Load image
        image = Image.open(image_path).convert("RGB")

        # Resize
        image = image.resize(self.input_shape)

        # Convert to array and normalize
        image_array = np.array(image, dtype=np.float32) / 255.0

        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)

        return image_array

    def predict(
        self, image_path: str, top_k: int = 3, generate_gradcam: bool = True
    ) -> Dict:
        """Predict disease from image.

        Args:
            image_path: Path to leaf image
            top_k: Number of top predictions to return
            generate_gradcam: Whether to generate Grad-CAM visualization

        Returns:
            Dictionary with prediction results
        """
        # Preprocess image
        image_array = self.preprocess_image(image_path)

        # Make prediction
        predictions = self.model.predict(image_array, verbose=0)[0]

        # Get top k predictions
        top_indices = np.argsort(predictions)[-top_k:][::-1]

        # Prepare results
        results = {
            "image_path": image_path,
            "disease": self.class_names[top_indices[0]],
            "confidence": float(predictions[top_indices[0]]),
            "top_predictions": [],
        }

        # Add top k predictions
        for idx in top_indices:
            results["top_predictions"].append(
                {"class": self.class_names[idx], "confidence": float(predictions[idx])}
            )

        # Get treatment recommendation
        results["treatment"] = get_treatment_recommendation(results["disease"])

        # Generate Grad-CAM if requested
        if generate_gradcam:
            gradcam_path = self._generate_gradcam(image_path, top_indices[0])
            results["gradcam_path"] = str(gradcam_path)

        return results

    def _generate_gradcam(self, image_path: str, class_idx: int) -> Path:
        """Generate Grad-CAM visualization."""
        # Get output path
        output_dir = Path("outputs/gradcam")
        output_dir.mkdir(parents=True, exist_ok=True)

        image_name = Path(image_path).stem
        output_path = output_dir / f"{image_name}_gradcam.png"

        # Generate Grad-CAM
        self.gradcam.generate_heatmap(image_path, class_idx, output_path)

        return output_path

    def predict_batch(self, image_paths: List[str]) -> List[Dict]:
        """Predict diseases for multiple images."""
        results = []

        for image_path in image_paths:
            try:
                result = self.predict(image_path, generate_gradcam=False)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                results.append({"image_path": image_path, "error": str(e)})

        return results
