"""Setup configuration for Plant Disease Detection System."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="plant-disease-detection",
    version="1.0.0",
    author="Priyanka Bolem",
    author_email="priyankabolem@gmail.com",
    description="A deep learning system for detecting plant diseases from leaf images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/priyankabolem/PlantDiseasesDetection",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "plant-disease-detect=src.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.h5", "*.yaml", "*.yml"],
    },
)