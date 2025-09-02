"""Test model loading and predictions."""

import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import tensorflow as tf  # noqa: E402
from src.models.architectures import ModelBuilder  # noqa: E402

# Ensure TensorFlow is working
tf.config.set_visible_devices([], 'GPU')  # Force CPU for tests


class TestModel:
    """Test model functionality."""

    def test_model_build(self):
        """Test model building."""
        builder = ModelBuilder(num_classes=38, input_shape=(224, 224, 3))
        model = builder.build_model(architecture="custom-cnn", pretrained=False)

        assert model is not None
        assert len(model.layers) > 0
        assert model.output_shape == (None, 38)

    def test_model_prediction_shape(self):
        """Test model prediction output shape."""
        builder = ModelBuilder(num_classes=38, input_shape=(224, 224, 3))
        model = builder.build_model(architecture="custom-cnn", pretrained=False)

        # Test with batch of images
        test_input = np.random.random((5, 224, 224, 3))
        predictions = model.predict(test_input, verbose=0)

        assert predictions.shape == (5, 38)
        assert np.allclose(predictions.sum(axis=1), 1.0)  # Softmax sums to 1

    def test_model_weights_loading(self):
        """Test loading pre-trained weights."""
        model_path = Path("weights/pretrained/best_model.h5")

        if model_path.exists():
            builder = ModelBuilder(num_classes=38, input_shape=(224, 224, 3))
            model = builder.build_model(architecture="custom-cnn", pretrained=False)

            # Should not raise exception
            model.load_weights(str(model_path))

            # Check that weights were actually loaded (at least one layer has non-zero weights)
            weights_loaded = False
            for layer in model.layers:
                if hasattr(layer, "weights") and layer.weights:
                    layer_weights = layer.get_weights()
                    if layer_weights and not np.allclose(layer_weights[0], 0):
                        weights_loaded = True
                        break

            assert weights_loaded, "No non-zero weights found after loading"

    def test_different_architectures(self):
        """Test building different architectures."""
        architectures = ["custom-cnn", "resnet50", "efficientnet-b0", "mobilenet-v2"]

        for arch in architectures:
            builder = ModelBuilder(num_classes=38, input_shape=(224, 224, 3))
            model = builder.build_model(architecture=arch, pretrained=False)

            assert model is not None
            assert model.output_shape == (None, 38)
