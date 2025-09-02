# Setting up GitHub as Primary Source for Hugging Face Space

This guide explains how to configure Hugging Face Space to automatically sync from GitHub repository.

## Prerequisites

- GitHub repository: https://github.com/priyankabolem/PlantDiseasesDetection.git
- Hugging Face Space: https://huggingface.co/spaces/Priyabolem/plant-disease-detection

## Steps to Set Up GitHub Sync

### 1. Configure Space Settings

Navigate to Hugging Face Space settings:

1. Go to: https://huggingface.co/spaces/Priyabolem/plant-disease-detection/settings
2. Scroll to "Sync with GitHub" section
3. Click "Link to GitHub repository"

### 2. Repository Configuration

Enter the following details:
- **Repository URL**: `https://github.com/priyankabolem/PlantDiseasesDetection.git`
- **Branch**: `main`
- **Sync direction**: GitHub → Space (one-way sync)

### 3. Authentication

Required steps:
1. Authorize Hugging Face to access GitHub repository
2. Grant read permissions to the repository
3. Optionally set up webhook for automatic updates

### 4. Sync Configuration

Create `.github/workflows/sync-to-hf.yml` in GitHub repository:

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
          git remote add hf https://huggingface.co/spaces/Priyabolem/plant-disease-detection
          git push --force hf main
```

### 5. Required Secrets

Add these secrets to GitHub repository:
- `HF_TOKEN`: Hugging Face access token with write permissions

### 6. Manual Sync Commands

For manual syncing:

```bash
# Add HF Space as remote
git remote add hf https://huggingface.co/spaces/Priyabolem/plant-disease-detection

# Push to both GitHub and HF
git push origin main
git push hf main
```

## Verification

After setup, verify the sync:
1. Make a change to GitHub repository
2. Check if change appears in HF Space
3. Monitor Space logs for sync errors

## Troubleshooting

- **Authentication errors**: Regenerate HF token
- **LFS issues**: Ensure Git LFS is installed and configured
- **Build failures**: Check Space logs at https://huggingface.co/spaces/Priyabolem/plant-disease-detection/logs

## Current Status

The model weights and fixes have been deployed to Hugging Face Space. To sync changes to GitHub repository:

1. Pull latest changes from HF Space
2. Push to GitHub repository
3. Set up automatic sync as described above