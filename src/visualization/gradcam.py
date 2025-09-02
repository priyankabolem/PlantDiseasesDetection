"""Grad-CAM visualization for model interpretability."""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


class GradCAM:
    """Gradient-weighted Class Activation Mapping."""

    def __init__(self, model: tf.keras.Model, layer_name: str = None):
        """Initialize Grad-CAM.

        Args:
            model: Trained Keras model
            layer_name: Name of convolutional layer to visualize.
                       If None, uses the last convolutional layer
        """
        self.model = model
        self.layer_name = layer_name or self._find_target_layer()

        # Create model to get activations and gradients
        self.grad_model = self._build_grad_model()

    def _find_target_layer(self) -> str:
        """Find the last convolutional layer in the model."""
        for layer in reversed(self.model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer.name

        raise ValueError("No Conv2D layer found in the model")

    def _build_grad_model(self) -> Model:
        """Build model that outputs activations and predictions."""
        # Get the convolutional layer
        conv_layer = self.model.get_layer(self.layer_name)

        # Create a model that maps the input image to the activations
        # of the target layer and the output predictions
        grad_model = Model(
            inputs=self.model.inputs, outputs=[conv_layer.output, self.model.outputs]
        )

        return grad_model

    def generate_heatmap(
        self,
        image_path: str,
        class_idx: int,
        output_path: str = None,
        alpha: float = 0.4,
    ) -> np.ndarray:
        """Generate Grad-CAM heatmap.

        Args:
            image_path: Path to input image
            class_idx: Index of the class to visualize
            output_path: Path to save visualization (optional)
            alpha: Transparency for heatmap overlay

        Returns:
            Heatmap array
        """
        # Load and preprocess image
        img = tf.keras.preprocessing.image.load_img(
            image_path, target_size=self.model.input_shape[1:3]
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Generate heatmap
        heatmap = self._make_gradcam_heatmap(img_array, class_idx)

        # Create visualization
        if output_path:
            self._save_and_display_gradcam(
                image_path, heatmap, output_path, alpha=alpha
            )

        return heatmap

    def _make_gradcam_heatmap(
        self, img_array: np.ndarray, pred_index: int
    ) -> np.ndarray:
        """Generate Grad-CAM heatmap."""
        # Record operations for automatic differentiation
        with tf.GradientTape() as tape:
            # Get activations and predictions
            outputs = self.grad_model(img_array)
            conv_outputs, predictions = outputs[0], outputs[1]

            # Handle both single and multiple outputs
            if isinstance(predictions, list):
                predictions = predictions[0]

            # Get the score for the predicted class
            class_score = predictions[:, pred_index]

        # Calculate gradients of the predicted class with respect to
        # the output feature map of the target convolutional layer
        grads = tape.gradient(class_score, conv_outputs)

        # Check if gradients are None (can happen with simple models)
        if grads is None:
            # Create dummy heatmap if gradients are None
            conv_shape = conv_outputs.shape[1:3]
            return np.ones(conv_shape) * 0.5

        # Pool the gradients over all the axes leaving out the batch dimension
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Get the values of the last convolutional layer output and pooled gradients
        conv_outputs = conv_outputs[0].numpy()
        pooled_grads = pooled_grads.numpy()

        # Weight the channels by corresponding gradients
        for i in range(pooled_grads.shape[-1]):
            conv_outputs[:, :, i] *= pooled_grads[i]

        # Average the weighted activation channels
        heatmap = np.mean(conv_outputs, axis=-1)

        # Normalize the heatmap
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1

        return heatmap

    def _save_and_display_gradcam(
        self, img_path: str, heatmap: np.ndarray, output_path: str, alpha: float = 0.4
    ):
        """Save Grad-CAM visualization."""
        # Load the original image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize heatmap to match original image
        heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

        # Convert heatmap to RGB
        heatmap_colored = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Superimpose the heatmap on original image
        superimposed_img = heatmap_colored * alpha + img * (1 - alpha)
        superimposed_img = np.uint8(superimposed_img)

        # Create figure with original and Grad-CAM images
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        axes[0].imshow(img)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        # Heatmap
        im = axes[1].imshow(heatmap_resized, cmap="jet")
        axes[1].set_title("Grad-CAM Heatmap")
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        # Superimposed
        axes[2].imshow(superimposed_img)
        axes[2].set_title("Grad-CAM Overlay")
        axes[2].axis("off")

        plt.tight_layout()

        # Save figure
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved Grad-CAM visualization to {output_path}")

    def generate_batch_heatmaps(
        self, image_paths: list, class_indices: list, output_dir: str
    ):
        """Generate Grad-CAM heatmaps for multiple images."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for img_path, class_idx in zip(image_paths, class_indices):
            img_name = Path(img_path).stem
            output_path = output_dir / f"{img_name}_gradcam.png"

            try:
                self.generate_heatmap(img_path, class_idx, str(output_path))
            except Exception as e:
                logger.error(f"Failed to generate Grad-CAM for {img_path}: {e}")


def visualize_feature_maps(
    model: tf.keras.Model, image_path: str, layer_names: list, output_path: str
):
    """Visualize intermediate feature maps.

    Args:
        model: Trained model
        image_path: Path to input image
        layer_names: List of layer names to visualize
        output_path: Path to save visualization
    """
    # Load and preprocess image
    img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=model.input_shape[1:3]
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Create model to extract features
    outputs = [model.get_layer(name).output for name in layer_names]
    feature_model = Model(inputs=model.input, outputs=outputs)

    # Get feature maps
    features = feature_model.predict(img_array)

    # Plot feature maps
    fig, axes = plt.subplots(len(layer_names), 8, figsize=(20, 3 * len(layer_names)))

    for i, (layer_name, feature_map) in enumerate(zip(layer_names, features)):
        # Select 8 feature maps
        n_features = min(8, feature_map.shape[-1])

        for j in range(n_features):
            ax = axes[i, j] if len(layer_names) > 1 else axes[j]
            ax.imshow(feature_map[0, :, :, j], cmap="viridis")
            ax.axis("off")

            if j == 0:
                ax.set_title(f"{layer_name}", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved feature maps to {output_path}")
