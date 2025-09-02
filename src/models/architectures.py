import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50, EfficientNetB0, MobileNetV2
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ModelBuilder:
    """Build different model architectures for plant disease classification."""
    
    def __init__(self, num_classes: int, input_shape: Tuple[int, int, int] = (224, 224, 3)):
        """Initialize model builder.
        
        Args:
            num_classes: Number of disease classes
            input_shape: Input image shape (height, width, channels)
        """
        self.num_classes = num_classes
        self.input_shape = input_shape
        
    def build_model(self, 
                   architecture: str = "efficientnet-b0",
                   pretrained: bool = True,
                   freeze_base: bool = False,
                   dropout_rate: float = 0.5) -> tf.keras.Model:
        """Build model with specified architecture.
        
        Args:
            architecture: Model architecture name
            pretrained: Use ImageNet pretrained weights
            freeze_base: Freeze base model layers
            dropout_rate: Dropout rate for regularization
            
        Returns:
            Compiled Keras model
        """
        architecture = architecture.lower()
        
        if architecture == "resnet50":
            model = self._build_resnet50(pretrained, freeze_base, dropout_rate)
        elif architecture == "efficientnet-b0":
            model = self._build_efficientnet(pretrained, freeze_base, dropout_rate)
        elif architecture == "mobilenet-v2":
            model = self._build_mobilenet(pretrained, freeze_base, dropout_rate)
        elif architecture == "custom-cnn":
            model = self._build_custom_cnn(dropout_rate)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
            
        logger.info(f"Built {architecture} model with {model.count_params():,} parameters")
        return model
    
    def _build_resnet50(self, pretrained: bool, freeze_base: bool, dropout_rate: float) -> tf.keras.Model:
        """Build ResNet50 model."""
        weights = "imagenet" if pretrained else None
        base_model = ResNet50(
            weights=weights,
            include_top=False,
            input_shape=self.input_shape
        )
        
        if freeze_base:
            base_model.trainable = False
            
        inputs = tf.keras.Input(shape=self.input_shape)
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = tf.keras.Model(inputs, outputs)
        return model
    
    def _build_efficientnet(self, pretrained: bool, freeze_base: bool, dropout_rate: float) -> tf.keras.Model:
        """Build EfficientNetB0 model."""
        weights = "imagenet" if pretrained else None
        base_model = EfficientNetB0(
            weights=weights,
            include_top=False,
            input_shape=self.input_shape
        )
        
        if freeze_base:
            base_model.trainable = False
            
        inputs = tf.keras.Input(shape=self.input_shape)
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = tf.keras.Model(inputs, outputs)
        return model
    
    def _build_mobilenet(self, pretrained: bool, freeze_base: bool, dropout_rate: float) -> tf.keras.Model:
        """Build MobileNetV2 model."""
        weights = "imagenet" if pretrained else None
        base_model = MobileNetV2(
            weights=weights,
            include_top=False,
            input_shape=self.input_shape
        )
        
        if freeze_base:
            base_model.trainable = False
            
        inputs = tf.keras.Input(shape=self.input_shape)
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = tf.keras.Model(inputs, outputs)
        return model
    
    def _build_custom_cnn(self, dropout_rate: float) -> tf.keras.Model:
        """Build custom CNN model matching the saved weights structure exactly."""
        # Match the exact architecture: conv2d_1 -> max_pooling2d -> conv2d_2 -> global_avg_pool -> predictions
        inputs = tf.keras.Input(shape=self.input_shape)
        x = layers.Conv2D(32, (3, 3), activation='relu', name='conv2d_1')(inputs)
        x = layers.MaxPooling2D(2, 2, name='max_pooling2d')(x)  
        x = layers.Conv2D(64, (3, 3), activation='relu', name='conv2d_2')(x)
        x = layers.GlobalAveragePooling2D(name='global_average_pooling2d')(x)
        outputs = layers.Dense(self.num_classes, activation='softmax', name='predictions')(x)
        
        model = tf.keras.Model(inputs, outputs)
        return model
    
    def _build_simple_cnn(self, dropout_rate: float) -> tf.keras.Model:
        """Build very simple CNN that might match saved weights."""
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        return model
    
    def unfreeze_layers(self, model: tf.keras.Model, num_layers: int = 20) -> tf.keras.Model:
        """Unfreeze top layers for fine-tuning.
        
        Args:
            model: Keras model
            num_layers: Number of layers to unfreeze from the top
            
        Returns:
            Model with unfrozen layers
        """
        # Find base model
        base_model = None
        for layer in model.layers:
            if isinstance(layer, (ResNet50, EfficientNetB0, MobileNetV2)):
                base_model = layer
                break
                
        if base_model:
            base_model.trainable = True
            # Freeze all layers except the last num_layers
            for layer in base_model.layers[:-num_layers]:
                layer.trainable = False
                
            logger.info(f"Unfroze top {num_layers} layers of base model")
            
        return model