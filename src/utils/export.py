import tensorflow as tf
import onnx
import tf2onnx
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def export_model(model: tf.keras.Model, output_dir: Path, format: str):
    """Export model in specified format.
    
    Args:
        model: Keras model to export
        output_dir: Directory to save exported model
        format: Export format ('h5', 'savedmodel', 'onnx', 'tflite')
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if format == 'h5':
        model.save(output_dir / 'model.h5')
        logger.info(f"Exported model in H5 format to {output_dir / 'model.h5'}")
        
    elif format == 'savedmodel':
        model.save(output_dir / 'savedmodel')
        logger.info(f"Exported model in SavedModel format to {output_dir / 'savedmodel'}")
        
    elif format == 'onnx':
        export_to_onnx(model, output_dir / 'model.onnx')
        
    elif format == 'tflite':
        export_to_tflite(model, output_dir / 'model.tflite')
        
    else:
        raise ValueError(f"Unknown export format: {format}")


def export_to_onnx(model: tf.keras.Model, output_path: Path):
    """Export model to ONNX format."""
    try:
        # Get model input shape
        input_signature = [tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype)]
        
        # Convert to ONNX
        onnx_model, _ = tf2onnx.convert.from_keras(
            model,
            input_signature=input_signature,
            opset=13
        )
        
        # Save ONNX model
        onnx.save(onnx_model, str(output_path))
        logger.info(f"Exported model in ONNX format to {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to export to ONNX: {e}")
        raise


def export_to_tflite(model: tf.keras.Model, output_path: Path, quantize: bool = False):
    """Export model to TFLite format."""
    try:
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        if quantize:
            # Apply post-training quantization
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
        tflite_model = converter.convert()
        
        # Save TFLite model
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
            
        logger.info(f"Exported model in TFLite format to {output_path}")
        
        # Print model size
        model_size = output_path.stat().st_size / 1024 / 1024
        logger.info(f"TFLite model size: {model_size:.2f} MB")
        
    except Exception as e:
        logger.error(f"Failed to export to TFLite: {e}")
        raise


def verify_onnx_model(onnx_path: Path):
    """Verify ONNX model is valid."""
    try:
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX model verification passed")
        return True
    except Exception as e:
        logger.error(f"ONNX model verification failed: {e}")
        return False


def benchmark_tflite_model(tflite_path: Path, test_image_path: Path):
    """Benchmark TFLite model inference speed."""
    import time
    import numpy as np
    from PIL import Image
    
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Prepare test image
    input_shape = input_details[0]['shape'][1:3]
    image = Image.open(test_image_path).convert('RGB')
    image = image.resize(input_shape)
    image_array = np.array(image, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    
    # Benchmark
    num_runs = 100
    start_time = time.time()
    
    for _ in range(num_runs):
        interpreter.set_tensor(input_details[0]['index'], image_array)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
    
    avg_time = (time.time() - start_time) / num_runs * 1000
    logger.info(f"Average inference time: {avg_time:.2f} ms")
    
    return avg_time