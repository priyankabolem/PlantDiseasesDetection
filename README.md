---
title: Plant Disease Detection
colorFrom: green
colorTo: yellow
sdk: streamlit
sdk_version: "1.36.0"
app_file: app_showcase.py
pinned: false
---

# Plant Disease Detection System

[![Hugging Face Space](https://img.shields.io/badge/Hugging%20Face-Space-yellow)](https://huggingface.co/spaces/Priyabolem/plant-disease-detection)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.13+](https://img.shields.io/badge/tensorflow-2.13+-orange.svg)](https://www.tensorflow.org/)

A production-ready deep learning system for detecting plant diseases from leaf images. Features a custom CNN architecture trained on 38 disease categories across 14 plant species. Now with realistic sample images showing actual disease symptoms.

> **Note**: This demo showcases the system's functionality and architecture. For accurate disease predictions, the model requires training on a comprehensive plant disease dataset. See [Model Status](MODEL_STATUS.md) for details.

## Live Demo

- **[Hugging Face Space](https://huggingface.co/spaces/Priyabolem/plant-disease-detection)** - Interactive Streamlit application
- **[API Documentation](https://plant-disease-api.onrender.com/docs)** - FastAPI backend (Coming Soon)

## Features

- **Real-time Disease Detection**: Upload a leaf image and get instant disease diagnosis
- **38 Disease Categories**: Covers 14 different plant species including tomatoes, apples, grapes, and more
- **Treatment Recommendations**: Get actionable treatment advice for detected diseases
- **Confidence Scores**: See prediction confidence levels for informed decision-making
- **Grad-CAM Visualizations**: Understand what the model is looking at (optional)
- **Mobile Responsive**: Works on desktop and mobile devices

## Screenshots

![App Demo](assets/demo_screenshot.png)

## Technical Stack

- **Frontend**: Streamlit with custom CSS styling
- **Backend**: FastAPI (optional API deployment)
- **ML Framework**: TensorFlow/Keras
- **Model Architecture**: Custom CNN with GlobalAveragePooling
- **Deployment**: Hugging Face Spaces + Render
- **Version Control**: GitHub with Git LFS for model weights

## Model Performance

*Target performance with properly trained model:*

| Metric | Value |
|--------|-------|
| Validation Accuracy | 92.5% |
| Top-3 Accuracy | 98.2% |
| Inference Time | ~50ms |
| Model Size | 116KB |

## Quick Start

### Local Development

1. **Clone the repository**
```bash
git clone https://github.com/priyankabolem/PlantDiseasesDetection.git
cd plant-disease-detection
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the Streamlit app**
```bash
streamlit run app_showcase.py
```

4. **Access the app**
Open browser to `http://localhost:8501`

### Using Docker

```bash
docker build -t plant-disease-app .
docker run -p 8501:8501 plant-disease-app
```

## Supported Plants and Diseases

The model can detect diseases in the following plants:

- **Apple**: Apple scab, Black rot, Cedar apple rust, Healthy
- **Blueberry**: Healthy
- **Cherry**: Powdery mildew, Healthy
- **Corn**: Cercospora leaf spot, Common rust, Northern Leaf Blight, Healthy
- **Grape**: Black rot, Esca, Leaf blight, Healthy
- **Orange**: Haunglongbing (Citrus greening)
- **Peach**: Bacterial spot, Healthy
- **Pepper**: Bacterial spot, Healthy
- **Potato**: Early blight, Late blight, Healthy
- **Strawberry**: Leaf scorch, Healthy
- **Tomato**: Bacterial spot, Early blight, Late blight, Leaf Mold, and more

## Project Structure

```
plant-disease-detection/
├── app_showcase.py          # Main Streamlit application
├── requirements.txt         # Python dependencies
├── train_model.py          # Model training script
├── src/                    # Source code modules
│   ├── models/            # Model architectures
│   ├── data/              # Data loading and preprocessing
│   ├── inference.py       # Inference pipeline
│   └── visualization/     # Grad-CAM and plotting
├── weights/               # Model weights (Git LFS)
│   └── pretrained/       
│       ├── best_model.h5
│       └── class_names.json
├── sample_images/         # Example images for testing
├── scripts/              # Utility scripts
└── tests/               # Unit tests
```

## Development

### Training a New Model

```bash
python train_model.py --data-dir /path/to/dataset --epochs 50 --batch-size 32
```

### Running Tests

```bash
pytest tests/ -v
```

### Code Quality

```bash
# Format code
black .

# Lint
flake8 .

# Type checking
mypy .
```

## Deployment

### Hugging Face Spaces (Primary)

The app automatically deploys to Hugging Face Spaces when pushing to the main branch.

1. Fork this repository
2. Create a new Space on Hugging Face
3. Connect GitHub repository
4. The Space will auto-build and deploy

### Render API (Secondary)

Deploy the FastAPI backend:

1. Create a new Web Service on Render
2. Connect GitHub repository
3. Set build command: `pip install -r requirements-api.txt`
4. Set start command: `uvicorn api:app --host 0.0.0.0 --port $PORT`

## Contributing

Contributions are welcome. Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Dataset inspiration from PlantVillage
- Built with TensorFlow and Streamlit
- Deployed on Hugging Face Spaces

## Author

**Priyanka Bolem**
- Email: priyankabolem@gmail.com
- LinkedIn: [Priyanka Bolem](https://www.linkedin.com/in/priyanka-bolem/)
- GitHub: [@priyankabolem](https://github.com/priyankabolem/PlantDiseasesDetection)

## Troubleshooting

### Common Issues

1. **Low confidence predictions**: Ensure using good quality, well-lit leaf images
2. **Slow loading**: First load may take time as the model initializes
3. **Memory errors**: Try reducing batch size or image resolution

### Getting Help

- Open an issue on GitHub
- Check the [FAQ](docs/FAQ.md)
- Contact: priyankabolem@gmail.com

---

Made by Priyanka Bolem