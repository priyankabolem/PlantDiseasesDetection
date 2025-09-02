# 🚀 Deployment Guide

This guide covers deploying the Plant Disease Detection System to various platforms.

## 📋 Prerequisites

- GitHub account with repository set up
- Hugging Face account (for Spaces deployment)
- Render account (optional, for API deployment)
- Git with LFS installed locally

## 🤗 Hugging Face Spaces Deployment

### Option 1: Direct GitHub Integration (Recommended)

1. **Fork/Clone this repository to your GitHub**
   ```bash
   git clone https://github.com/yourusername/plant-disease-detection.git
   cd plant-disease-detection
   ```

2. **Create a new Space on Hugging Face**
   - Go to [Hugging Face Spaces](https://huggingface.co/new-space)
   - Name your Space (e.g., `plant-disease-detection`)
   - Select **Streamlit** as the SDK
   - Choose **Public** visibility

3. **Connect to GitHub**
   - In your Space settings, go to "Files and versions"
   - Click "Link a GitHub repository"
   - Select your repository
   - Choose the branch (usually `main`)

4. **Configure automatic sync**
   - Enable "Sync with GitHub repository"
   - The Space will automatically rebuild when you push to GitHub

### Option 2: Direct Upload

1. **Clone the repository locally**
   ```bash
   git clone https://github.com/yourusername/plant-disease-detection.git
   cd plant-disease-detection
   ```

2. **Install Hugging Face CLI**
   ```bash
   pip install huggingface-hub
   huggingface-cli login
   ```

3. **Create and push to Space**
   ```bash
   # Add Hugging Face remote
   git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/plant-disease-detection
   
   # Push to Hugging Face
   git push hf main
   ```

## 🌐 GitHub Pages (Documentation)

1. **Enable GitHub Pages**
   - Go to Settings → Pages in your repository
   - Select source: "Deploy from a branch"
   - Choose branch: `main` and folder: `/docs`

2. **Access your documentation**
   - URL: `https://YOUR_USERNAME.github.io/plant-disease-detection/`

## 🔧 Render Deployment (API)

1. **Create `render.yaml`**
   ```yaml
   services:
     - type: web
       name: plant-disease-api
       env: python
       buildCommand: "pip install -r requirements-api.txt"
       startCommand: "uvicorn api:app --host 0.0.0.0 --port $PORT"
       envVars:
         - key: PYTHON_VERSION
           value: 3.9
   ```

2. **Connect to Render**
   - Go to [Render Dashboard](https://dashboard.render.com/)
   - New → Web Service
   - Connect your GitHub repository
   - Render will auto-detect the `render.yaml`

3. **Deploy**
   - Click "Create Web Service"
   - Wait for build and deployment

## 🐳 Docker Deployment

1. **Build the image**
   ```bash
   docker build -t plant-disease-detector .
   ```

2. **Run locally**
   ```bash
   docker run -p 8501:8501 plant-disease-detector
   ```

3. **Push to Docker Hub**
   ```bash
   docker tag plant-disease-detector YOUR_USERNAME/plant-disease-detector
   docker push YOUR_USERNAME/plant-disease-detector
   ```

## ⚙️ Environment Variables

Configure these in your deployment platform:

```bash
# Hugging Face Spaces
HF_TOKEN=your_hugging_face_token  # For private models

# Render/Heroku
PORT=8501  # Usually set automatically
STREAMLIT_SERVER_PORT=$PORT
STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Optional
MODEL_CACHE_DIR=/app/models
LOG_LEVEL=INFO
```

## 🔄 CI/CD with GitHub Actions

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to Hugging Face

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
        with:
          lfs: true
      
      - name: Push to Hugging Face
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: |
          git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/plant-disease-detection
          git push hf main
```

## 📊 Monitoring

### Hugging Face Spaces
- View logs: Space settings → Logs
- Monitor usage: Space settings → Usage

### Render
- View logs: Dashboard → Your service → Logs
- Set up health checks in `render.yaml`

## 🚨 Troubleshooting

### Model Loading Issues
```bash
# Ensure Git LFS is tracking model files
git lfs track "*.h5"
git add .gitattributes
git commit -m "Track model files with LFS"
```

### Memory Issues
- Reduce model size or use quantization
- Increase dyno/container memory limits
- Implement model caching

### Slow Startup
- Use `@st.cache_resource` for model loading
- Implement health checks with longer timeouts
- Consider using smaller model variants

## 📝 Deployment Checklist

- [ ] Model files tracked with Git LFS
- [ ] Environment variables configured
- [ ] Requirements files up to date
- [ ] README includes deployment badges
- [ ] Health check endpoint implemented
- [ ] Logging configured properly
- [ ] Error handling for edge cases
- [ ] CORS settings (for API)
- [ ] Rate limiting configured
- [ ] SSL/HTTPS enabled

## 🔗 Useful Links

- [Streamlit Deployment Docs](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app)
- [Hugging Face Spaces Docs](https://huggingface.co/docs/hub/spaces)
- [Render Docs](https://render.com/docs)
- [Git LFS Documentation](https://git-lfs.github.com/)

---

For questions or issues, contact: priyankabolem@gmail.com