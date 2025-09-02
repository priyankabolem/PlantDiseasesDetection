# 🔗 GitHub Repository Setup Guide

This guide will help you set up a proper GitHub repository as the source of truth for your project.

## 📋 Steps to Set Up GitHub Repository

### 1. Create GitHub Repository

1. Go to [GitHub](https://github.com/new)
2. Create a new repository:
   - Name: `plant-disease-detection`
   - Description: "AI-powered plant disease detection from leaf images"
   - Visibility: Public
   - Do NOT initialize with README (we already have one)

### 2. Set Up Local Repository

```bash
# Add GitHub as a new remote
git remote add github https://github.com/YOUR_USERNAME/plant-disease-detection.git

# View all remotes
git remote -v

# Push to GitHub
git push github main
```

### 3. Configure Hugging Face to Sync from GitHub

1. Go to your [Hugging Face Space Settings](https://huggingface.co/spaces/priyabolem/plant-disease-detection/settings)
2. Under "Repository settings":
   - Click "Link a GitHub repository"
   - Authorize Hugging Face to access your GitHub
   - Select your `plant-disease-detection` repository
   - Choose branch: `main`
   - Enable "Sync with GitHub repository"

### 4. Set Up GitHub Secrets (for CI/CD)

In your GitHub repository settings → Secrets and variables → Actions:

```
HF_TOKEN: Your Hugging Face write token
HF_USERNAME: priyabolem
DOCKER_USERNAME: (optional) Your Docker Hub username
DOCKER_PASSWORD: (optional) Your Docker Hub password
```

### 5. Update Remote URLs

After setting up GitHub as primary:

```bash
# Make GitHub the primary (origin)
git remote rename origin huggingface
git remote rename github origin

# Verify
git remote -v
# Should show:
# origin    https://github.com/YOUR_USERNAME/plant-disease-detection.git (fetch)
# origin    https://github.com/YOUR_USERNAME/plant-disease-detection.git (push)
# huggingface    https://huggingface.co/spaces/priyabolem/plant-disease-detection (fetch)
# huggingface    https://huggingface.co/spaces/priyabolem/plant-disease-detection (push)
```

## 🔄 Workflow After Setup

### Development Workflow

1. **Make changes locally**
2. **Push to GitHub** (primary):
   ```bash
   git add .
   git commit -m "Your changes"
   git push origin main
   ```
3. **Auto-sync to Hugging Face** happens automatically
4. **CI/CD runs tests** via GitHub Actions

### Manual Push to Hugging Face (if needed)

```bash
git push huggingface main
```

## 🎯 Best Practices

1. **Always push to GitHub first** - it's your source of truth
2. **Let Hugging Face sync automatically** from GitHub
3. **Use GitHub Issues** for tracking bugs and features
4. **Create Pull Requests** for major changes
5. **Tag releases** for version control:
   ```bash
   git tag -a v1.0.0 -m "First stable release"
   git push origin v1.0.0
   ```

## 🚀 Quick Commands Reference

```bash
# Status check
git status
git remote -v

# Push to GitHub (primary)
git push origin main

# Push to Hugging Face (backup)
git push huggingface main

# Push to both
git push origin main && git push huggingface main

# Pull latest
git pull origin main

# Create feature branch
git checkout -b feature/new-feature
git push -u origin feature/new-feature
```

## 📊 Repository Structure

```
GitHub (Primary) ─────sync────→ Hugging Face Space
    ↓                              ↓
CI/CD Tests                   Live Demo
    ↓
Docker Hub (optional)
```

## 🔗 Useful Links

- [GitHub Repository](https://github.com/YOUR_USERNAME/plant-disease-detection)
- [Hugging Face Space](https://huggingface.co/spaces/priyabolem/plant-disease-detection)
- [GitHub Actions](https://github.com/YOUR_USERNAME/plant-disease-detection/actions)
- [Issues](https://github.com/YOUR_USERNAME/plant-disease-detection/issues)

---

For questions: priyankabolem@gmail.com