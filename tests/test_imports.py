"""Test basic imports to diagnose CI/CD issues."""

import sys
print(f"Python version: {sys.version}")
print(f"Python path: {sys.path}")

try:
    import numpy as np
    print(f"NumPy version: {np.__version__}")
except Exception as e:
    print(f"NumPy import error: {e}")
    sys.exit(1)

try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
except Exception as e:
    print(f"TensorFlow import error: {e}")
    sys.exit(1)

try:
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from src.models.architectures import ModelBuilder
    print("ModelBuilder import successful")
except Exception as e:
    print(f"ModelBuilder import error: {e}")
    sys.exit(1)

print("All imports successful!")