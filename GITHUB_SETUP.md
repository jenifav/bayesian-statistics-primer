# GitHub Setup Guide

## Quick Start Commands

Here are the essential Git Bash commands to get your Bayesian Statistics Primer repository on GitHub:

### 1. Prerequisites
```bash
# Configure Git (if not already done)
git config --global user.email "jennifer.h.favaloro@gmail.com" 
git config --global user.name "Jennifer Favaloro"
```

### 2. Create GitHub Repository
1. Go to [GitHub](https://github.com) and create a new repository
2. Name: `bayesian-statistics-primer`
3. Description: `A comprehensive guide to Bayesian statistics for frequentist-trained statisticians`
4. Make it **Public**
5. **DO NOT** initialize with README, .gitignore, or license
6. Click "Create repository"

### 3. Initialize and Push
```bash
# Navigate to the project directory
cd "$USERPROFILE\OneDrive\GitHub\bayesian-statistics-primer"  

# Initialize Git repository
git init

# Set main as default branch
git branch -M main

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: Add comprehensive Bayesian statistics primer"

# Add remote origin (replace YOUR_USERNAME)
git remote add origin https://github.com/jenifav/bayesian-statistics-primer.git

# Push to GitHub
git push -u origin main
```

### 4. Verify
Visit your GitHub repository to confirm all files uploaded correctly.

## Common Git Commands

### Daily Workflow
```bash
# Check status
git status

# Add changes
git add .

# Commit changes
git commit -m "Description of changes"

# Push to GitHub
git push
```

### Branch Management
```bash
# Create new branch
git checkout -b feature-name

# Switch branches
git checkout main

# Merge branch
git merge feature-name

# Delete branch
git branch -d feature-name
```

### Collaboration
```bash
# Pull latest changes
git pull

# View commit history
git log --oneline

# View differences
git diff
```

## Troubleshooting

### Authentication Issues
- Use personal access token instead of password
- Set up SSH keys for easier authentication
- Verify your username and repository name

### Repository Issues
```bash
# Change remote URL if needed
git remote set-url origin https://github.com/jenifav/bayesian-statistics-primer.git

# View current remote
git remote -v
```

### Merge Conflicts
```bash
# View conflicted files
git status

# After resolving conflicts
git add .
git commit -m "Resolve merge conflicts"
```

## Repository Structure

```
bayesian-statistics-primer/
├── README.md                    # Project overview
├── LICENSE                      # MIT license
├── CONTRIBUTING.md              # Contribution guidelines
├── .gitignore                   # Git ignore rules
├── github_workflow.sh           # This setup script
├── GITHUB_SETUP.md             # This guide
├── docs/
│   ├── bayesian_primer.md       # Main primer (25,000+ words)
│   └── multi_platform_guide.md  # Implementation guide
├── code/
│   ├── python/
│   │   └── bayesian_examples.py # Python implementation
│   ├── r/
│   │   └── bayesian_examples.R  # R implementation
│   └── sas/
│       └── bayesian_examples.sas # SAS implementation
├── images/
│   ├── bayesian_proportion_example.png
│   ├── bayesian_ab_test.png
│   └── bayesian_regression.png
└── examples/                    # Additional examples
```

## Next Steps

1. **Customize README**: Update the README with your specific information
2. **Add Examples**: Consider adding more examples in the `examples/` directory
3. **Enable Issues**: Turn on GitHub Issues for community feedback
4. **Add Topics**: Add relevant topics/tags to your repository
5. **Create Releases**: Tag important versions of your primer

## Sharing Your Work

- Share the repository URL with colleagues
- Consider submitting to relevant communities
- Add the repository to your professional profiles
- Write blog posts about your primer

---

**Happy Git-ing! 🚀**

