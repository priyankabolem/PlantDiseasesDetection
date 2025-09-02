#!/bin/bash

# Script to sync Hugging Face Space improvements to GitHub repository
# Usage: ./sync_to_github.sh

echo "Syncing Plant Disease Detection improvements to GitHub..."

# Configuration
GITHUB_REPO="https://github.com/priyankabolem/PlantDiseasesDetection.git"
TEMP_DIR="temp_github_sync"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Create temporary directory
echo -e "${YELLOW}Creating temporary directory...${NC}"
mkdir -p $TEMP_DIR
cd $TEMP_DIR

# Clone GitHub repository
echo -e "${YELLOW}Cloning GitHub repository...${NC}"
git clone $GITHUB_REPO github_repo
cd github_repo

# Add HF Space as remote
echo -e "${YELLOW}Adding Hugging Face Space as remote...${NC}"
git remote add hf https://huggingface.co/spaces/priyabolem/plant-disease-detection

# Fetch latest from HF Space
echo -e "${YELLOW}Fetching latest changes from HF Space...${NC}"
git fetch hf

# Create a new branch for the improvements
BRANCH_NAME="feature/hf-space-improvements-$(date +%Y%m%d)"
git checkout -b $BRANCH_NAME

# Key files to sync
echo -e "${YELLOW}Syncing key improvements...${NC}"

# Copy improved files from HF Space
FILES_TO_SYNC=(
    "weights/pretrained/best_model.h5"
    "src/models/architectures.py"
    "app_showcase.py"
    "requirements.txt"
    ".gitattributes"
    "scripts/create_realistic_weights.py"
)

for file in "${FILES_TO_SYNC[@]}"; do
    echo -e "  Syncing: $file"
    git checkout hf/main -- "$file" 2>/dev/null || echo -e "${RED}    File not found in HF Space${NC}"
done

# Add and commit changes
echo -e "${YELLOW}Committing improvements...${NC}"
git add .
git commit -m "feat: sync model improvements from Hugging Face Space

- Fixed model weights with proper feature detection
- Updated CNN architecture to match saved weights
- Improved confidence thresholds and error handling
- Added Git LFS support for model weights
- Enhanced model loading with TensorFlow compatibility

These changes fix the issue where all predictions had identical low confidence."

# Push to GitHub
echo -e "${YELLOW}Pushing to GitHub...${NC}"
git push origin $BRANCH_NAME

echo -e "${GREEN}Sync complete!${NC}"
echo -e "${GREEN}Branch created: $BRANCH_NAME${NC}"
echo -e "${GREEN}Next steps:${NC}"
echo "1. Go to: $GITHUB_REPO"
echo "2. Create a Pull Request from $BRANCH_NAME to main"
echo "3. Review and merge the improvements"

# Cleanup
cd ../..
rm -rf $TEMP_DIR

echo -e "${GREEN}Done! GitHub repository now has the working model.${NC}"