import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from typing import Tuple, Dict, Optional
import yaml
import logging

logger = logging.getLogger(__name__)


class PlantDiseaseDataLoader:
    """Data loader for plant disease dataset."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize data loader with configuration."""
        if config_path:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._default_config()

        self.image_size = tuple(self.config["data"]["image_size"])
        self.batch_size = self.config["data"]["batch_size"]
        self.validation_split = self.config["data"]["validation_split"]

    def _default_config(self) -> dict:
        """Return default configuration."""
        return {
            "data": {
                "image_size": [224, 224],
                "batch_size": 32,
                "validation_split": 0.2,
                "augmentation": {
                    "rotation_range": 20,
                    "width_shift_range": 0.2,
                    "height_shift_range": 0.2,
                    "horizontal_flip": True,
                    "zoom_range": 0.2,
                    "shear_range": 0.2,
                    "brightness_range": [0.8, 1.2],
                },
            }
        }

    def create_data_generators(
        self, train_dir: str, valid_dir: Optional[str] = None
    ) -> Tuple[
        tf.keras.preprocessing.image.DirectoryIterator,
        tf.keras.preprocessing.image.DirectoryIterator,
    ]:
        """Create training and validation data generators."""
        aug_config = self.config["data"]["augmentation"]

        # Training data generator with augmentation
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=aug_config["rotation_range"],
            width_shift_range=aug_config["width_shift_range"],
            height_shift_range=aug_config["height_shift_range"],
            horizontal_flip=aug_config["horizontal_flip"],
            zoom_range=aug_config["zoom_range"],
            shear_range=aug_config["shear_range"],
            brightness_range=aug_config["brightness_range"],
            validation_split=self.validation_split if valid_dir is None else 0,
        )

        # Validation data generator (no augmentation)
        valid_datagen = ImageDataGenerator(rescale=1.0 / 255)

        # Create generators
        if valid_dir is None:
            # Use validation split from training data
            train_generator = train_datagen.flow_from_directory(
                train_dir,
                target_size=self.image_size,
                batch_size=self.batch_size,
                class_mode="categorical",
                subset="training",
            )

            valid_generator = train_datagen.flow_from_directory(
                train_dir,
                target_size=self.image_size,
                batch_size=self.batch_size,
                class_mode="categorical",
                subset="validation",
            )
        else:
            # Use separate validation directory
            train_generator = train_datagen.flow_from_directory(
                train_dir,
                target_size=self.image_size,
                batch_size=self.batch_size,
                class_mode="categorical",
            )

            valid_generator = valid_datagen.flow_from_directory(
                valid_dir,
                target_size=self.image_size,
                batch_size=self.batch_size,
                class_mode="categorical",
            )

        logger.info(
            f"Found {train_generator.samples} training images belonging to {train_generator.num_classes} classes"
        )
        logger.info(
            f"Found {valid_generator.samples} validation images belonging to {valid_generator.num_classes} classes"
        )

        return train_generator, valid_generator

    def get_class_names(
        self, generator: tf.keras.preprocessing.image.DirectoryIterator
    ) -> Dict[int, str]:
        """Get class names from generator."""
        return {v: k for k, v in generator.class_indices.items()}

    def create_test_generator(
        self, test_dir: str
    ) -> tf.keras.preprocessing.image.DirectoryIterator:
        """Create test data generator."""
        test_datagen = ImageDataGenerator(rescale=1.0 / 255)

        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode="categorical",
            shuffle=False,
        )

        logger.info(
            f"Found {test_generator.samples} test images belonging to {test_generator.num_classes} classes"
        )

        return test_generator
