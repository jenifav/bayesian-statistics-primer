#!/bin/bash

# GitHub Workflow Script for Bayesian Statistics Primer
# This script provides step-by-step commands to set up and publish the repository to GitHub

echo "=================================================="
echo "GitHub Workflow for Bayesian Statistics Primer"
echo "=================================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}This script will guide you through setting up the Bayesian Statistics Primer repository on GitHub.${NC}"
echo ""

# Step 1: Prerequisites
echo -e "${YELLOW}STEP 1: Prerequisites${NC}"
echo "Before running this script, make sure you have:"
echo "1. Git installed on your system"
echo "2. A GitHub account"
echo "3. Git configured with your name and email"
echo ""
echo "To configure Git (if not already done):"
echo -e "${GREEN}git config --global user.name \"Your Name\"${NC}"
echo -e "${GREEN}git config --global user.email \"your.email@example.com\"${NC}"
echo ""

# Step 2: Create GitHub Repository
echo -e "${YELLOW}STEP 2: Create GitHub Repository${NC}"
echo "1. Go to https://github.com"
echo "2. Click the '+' button in the top right corner"
echo "3. Select 'New repository'"
echo "4. Repository name: bayesian-statistics-primer"
echo "5. Description: A comprehensive guide to Bayesian statistics for frequentist-trained statisticians"
echo "6. Make it Public (recommended for sharing)"
echo "7. DO NOT initialize with README, .gitignore, or license (we already have these)"
echo "8. Click 'Create repository'"
echo ""
echo "Press Enter when you've created the repository on GitHub..."
read -r

# Step 3: Initialize Local Repository
echo -e "${YELLOW}STEP 3: Initialize Local Repository${NC}"
echo "Navigate to the repository directory and initialize Git:"
echo ""
echo -e "${GREEN}cd bayesian-statistics-primer${NC}"
echo -e "${GREEN}git init${NC}"
echo -e "${GREEN}git branch -M main${NC}  # Use 'main' as the default branch"
echo ""

# Step 4: Add Files
echo -e "${YELLOW}STEP 4: Add Files to Repository${NC}"
echo "Add all files to the repository:"
echo ""
echo -e "${GREEN}git add .${NC}"
echo ""

# Step 5: Initial Commit
echo -e "${YELLOW}STEP 5: Create Initial Commit${NC}"
echo "Create your first commit:"
echo ""
echo -e "${GREEN}git commit -m \"Initial commit: Add comprehensive Bayesian statistics primer\"${NC}"
echo ""

# Step 6: Add Remote Origin
echo -e "${YELLOW}STEP 6: Add Remote Origin${NC}"
echo "Replace YOUR_USERNAME with your actual GitHub username:"
echo ""
echo -e "${GREEN}git remote add origin https://github.com/YOUR_USERNAME/bayesian-statistics-primer.git${NC}"
echo ""
echo "Alternative (if you have SSH set up):"
echo -e "${GREEN}git remote add origin git@github.com:YOUR_USERNAME/bayesian-statistics-primer.git${NC}"
echo ""

# Step 7: Push to GitHub
echo -e "${YELLOW}STEP 7: Push to GitHub${NC}"
echo "Push your code to GitHub:"
echo ""
echo -e "${GREEN}git push -u origin main${NC}"
echo ""

# Step 8: Verify
echo -e "${YELLOW}STEP 8: Verify Upload${NC}"
echo "Go to your GitHub repository page to verify all files were uploaded correctly."
echo "You should see:"
echo "- README.md with project description"
echo "- All code files in their respective directories"
echo "- Documentation files"
echo "- Images"
echo ""

# Additional Git Commands
echo -e "${YELLOW}USEFUL GIT COMMANDS FOR FUTURE UPDATES:${NC}"
echo ""
echo "Check repository status:"
echo -e "${GREEN}git status${NC}"
echo ""
echo "Add specific files:"
echo -e "${GREEN}git add filename.ext${NC}"
echo ""
echo "Add all changes:"
echo -e "${GREEN}git add .${NC}"
echo ""
echo "Commit changes:"
echo -e "${GREEN}git commit -m \"Description of changes\"${NC}"
echo ""
echo "Push changes:"
echo -e "${GREEN}git push${NC}"
echo ""
echo "Pull latest changes:"
echo -e "${GREEN}git pull${NC}"
echo ""
echo "View commit history:"
echo -e "${GREEN}git log --oneline${NC}"
echo ""
echo "Create a new branch:"
echo -e "${GREEN}git checkout -b new-feature-branch${NC}"
echo ""
echo "Switch branches:"
echo -e "${GREEN}git checkout main${NC}"
echo ""
echo "Merge branch:"
echo -e "${GREEN}git merge feature-branch${NC}"
echo ""

# Troubleshooting
echo -e "${YELLOW}TROUBLESHOOTING:${NC}"
echo ""
echo "If you get authentication errors:"
echo "1. Make sure you're using the correct username"
echo "2. Use a personal access token instead of password"
echo "3. Set up SSH keys for easier authentication"
echo ""
echo "If you get 'repository already exists' error:"
echo "1. Make sure the GitHub repository is empty"
echo "2. Or use: git remote set-url origin https://github.com/YOUR_USERNAME/bayesian-statistics-primer.git"
echo ""
echo "If you need to change the remote URL:"
echo -e "${GREEN}git remote set-url origin NEW_URL${NC}"
echo ""

# Best Practices
echo -e "${YELLOW}BEST PRACTICES:${NC}"
echo ""
echo "1. Always commit with descriptive messages"
echo "2. Pull before pushing if working with others"
echo "3. Use branches for new features"
echo "4. Keep commits focused and atomic"
echo "5. Test code before committing"
echo "6. Use .gitignore to exclude unnecessary files"
echo ""

echo -e "${GREEN}Setup complete! Your Bayesian Statistics Primer is ready for GitHub!${NC}"
echo ""
echo "Repository structure:"
echo "├── README.md                    # Project overview and instructions"
echo "├── LICENSE                      # MIT license"
echo "├── CONTRIBUTING.md              # Contribution guidelines"
echo "├── .gitignore                   # Git ignore rules"
echo "├── docs/"
echo "│   ├── bayesian_primer.md       # Main primer document"
echo "│   └── multi_platform_guide.md  # Implementation guide"
echo "├── code/"
echo "│   ├── python/                  # Python examples"
echo "│   ├── r/                       # R examples"
echo "│   └── sas/                     # SAS examples"
echo "├── images/                      # Generated plots and figures"
echo "└── examples/                    # Additional examples"
echo ""
echo -e "${BLUE}Happy coding and sharing your Bayesian knowledge! 🎉${NC}"

