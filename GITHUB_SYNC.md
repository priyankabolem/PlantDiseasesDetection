# Setting up GitHub as Primary Source for Hugging Face Space

This guide explains how to configure your Hugging Face Space to automatically sync from your GitHub repository.

## Prerequisites

- GitHub repository: https://github.com/priyankabolem/PlantDiseasesDetection.git
- Hugging Face Space: https://huggingface.co/spaces/priyabolem/plant-disease-detection

## Steps to Set Up GitHub Sync

### 1. Configure Space Settings

Go to your Hugging Face Space settings and add the following:

1. Navigate to: https://huggingface.co/spaces/priyabolem/plant-disease-detection/settings
2. Scroll to "Sync with GitHub" section
3. Click "Link to GitHub repository"

### 2. Repository Configuration

Enter the following details:
- **Repository URL**: `https://github.com/priyankabolem/PlantDiseasesDetection.git`
- **Branch**: `main` (or your default branch)
- **Sync direction**: GitHub → Space (one-way sync)

### 3. Authentication

You'll need to:
1. Authorize Hugging Face to access your GitHub repository
2. Grant read permissions to the repository
3. Optionally set up webhook for automatic updates

### 4. Sync Configuration

Create a `.github/workflows/sync-to-hf.yml` in your GitHub repo:

```yaml
name: Sync to Hugging Face

on:
  push:
    branches: [main]
  workflow_dispatch:

jobs:
  sync:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
        with:
          lfs: true
      
      - name: Push to Hugging Face
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: |
          git remote add hf https://huggingface.co/spaces/priyabolem/plant-disease-detection
          git push --force hf main
```

### 5. Required Secrets

Add these secrets to your GitHub repository:
- `HF_TOKEN`: Your Hugging Face access token with write permissions

### 6. Manual Sync Commands

If you prefer manual syncing:

```bash
# Add HF Space as remote
git remote add hf https://huggingface.co/spaces/priyabolem/plant-disease-detection

# Push to both GitHub and HF
git push origin main
git push hf main
```

## Verification

After setup, verify the sync is working:
1. Make a small change to your GitHub repo
2. Check if the change appears in your HF Space
3. Monitor the Space logs for any sync errors

## Troubleshooting

- **Authentication errors**: Regenerate your HF token
- **LFS issues**: Ensure Git LFS is installed and configured
- **Build failures**: Check Space logs at https://huggingface.co/spaces/priyabolem/plant-disease-detection/logs

## Current Status

The improved model weights and fixes have been pushed to the Hugging Face Space. To sync these changes to your GitHub repository:

1. Pull the latest changes from HF Space
2. Push to your GitHub repository
3. Set up the automatic sync as described above