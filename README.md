---
title: Plant Disease Detection (Recruiter Showcase)
emoji: 🌿
colorFrom: green
colorTo: indigo
sdk: streamlit
app_file: app_showcase.py
pinned: true
license: mit
---

# Plant Disease Detection — Interactive Demo

A fast, recruiter-friendly Streamlit app for detecting plant diseases from leaf images.  
**Accuracy:** 98.2% on a held-out test set across **38 classes** and **14 plant species**.  
**Goal:** < 30s cold start with cached model load, one-click samples, and mobile-ready layout.

---

## 🚀 Try It
- Click a **Sample** on the left sidebar (e.g., Tomato, Grape, Potato) for an instant demo.
- Or **Upload** your own JPG/PNG leaf image.
- The app returns the predicted disease, confidence, and brief care tips.

> Having trouble? Refresh the page — first run may take a bit longer while models load.

---

## ✨ Features
- One-click sample images for instant evaluation
- Drag-and-drop image upload
- Prediction with **confidence score**
- Concise **treatment recommendations**
- (Optional in app) Grad-CAM heatmap to visualize model attention
- Mobile-responsive UI

---

## 🧠 Model
- **Backbones:** ResNet50 / EfficientNetB0 / MobileNetV2 (selectable in code)
- **Input size:** 224×224
- **Classes:** 38 diseases across 14 crop species
- **Performance:** 98.2% accuracy (held-out test set)

---

## 📦 Tech
- **Streamlit** UI (`app_showcase.py`)
- **TensorFlow (CPU)** for inference
- Lightweight **requirements** for quick cold starts

---

## 🧪 How to Use
1. Choose a **Sample** or upload a leaf photo.
2. Click **Analyze** (if the UI requires it).
3. Read the predicted disease + confidence + suggested actions.
4. (Optional) Toggle Grad-CAM to see model focus regions.

---

## 🔌 API (Optional)
If you’re also running the FastAPI service, link to it here:
- **Swagger Docs:** https://plant-disease-api.onrender.com/docs
- Example:
  ```bash
  curl -X POST https://plant-disease-api.onrender.com/predict \
    -F "file=@sample.jpg"