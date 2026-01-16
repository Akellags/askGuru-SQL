# GitHub Workflow Guide for askGuru-SQL

**Complete steps to initialize, push, and manage this project on GitHub**

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Initial Setup (One-time)](#initial-setup-one-time)
3. [First Push](#first-push)
4. [Daily Workflow](#daily-workflow)
5. [Pulling Updates](#pulling-updates)
6. [Branching Strategy](#branching-strategy)
7. [Common Commands](#common-commands)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### 1. Git Installation

Verify git is installed:

```bash
git --version
```

If not installed:

```bash
# Ubuntu/Debian
sudo apt-get install git

# macOS
brew install git

# Windows
# Download from https://git-scm.com/download/win
```

### 2. GitHub Account

- Create account at https://github.com
- Verify email
- (Optional but recommended) Set up SSH key

### 3. Configure Git Locally

```bash
# Set your name
git config --global user.name "Your Name"

# Set your email
git config --global user.email "your.email@example.com"

# Verify
git config --global --list
```

### 4. Create .gitignore

Create file: `.gitignore` in project root

```bash
cat > .gitignore << 'EOF'
# Virtual environments
venv/
env/
.venv/

# Model weights (too large)
*.pt
*.pth
*.bin
models/
outputs/
checkpoints/

# Data (keep structure, ignore large files)
data/**/*.pkl
data/**/*.csv
*.jsonl
*.npy
*.npz

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Jupyter
.ipynb_checkpoints/
*.ipynb

# Logs
logs/
*.log
wandb/

# Cache
.cache/
.pytest_cache/
.coverage

# Environment
.env
.env.local
EOF

cat .gitignore
```

---

## Initial Setup (One-time)

### Step 1: Initialize Local Repository

```bash
cd /path/to/askGuru-SQL

# Initialize git
git init

# Verify
git status
```

Expected output:
```
On branch master

No commits yet

Untracked files:
  (use "git add <file>..." to include in what will be committed)
        ...files...
```

### Step 2: Create GitHub Repository

1. Go to https://github.com/new
2. **Repository name**: `askGuru-SQL`
3. **Description**: `Oracle EBS NL2SQL: LLaMA-3.3-70B + SQLCoder-70B Fine-tuning Framework`
4. **Visibility**: Private (recommended for proprietary code)
5. **Initialize with**: 
   - ❌ Do NOT add README (we have one)
   - ❌ Do NOT add .gitignore (we created one)
   - ❌ Do NOT add license (unless needed)
6. Click **Create repository**

### Step 3: Connect Local to Remote

You'll see instructions like:

```bash
# Add remote
git remote add origin https://github.com/Akellags/askGuru-SQL.git

# Verify
git remote -v
```

Expected output:
```
origin  https://github.com/Akellags/askGuru-SQL.git (fetch)
origin  https://github.com/Akellags/askGuru-SQL.git (push)
```

### Step 4: Add Files

```bash
# Stage all files
git add .

# Or stage specific files
git add DEPLOY_FINETUNE_UBUNTU24.md
git add DEPLOY_SQLCODER_UBUNTU24.md
git add custom_oracle_sqlcoder/

# Verify staging
git status
```

Expected output:
```
On branch master

No commits yet

Changes to be committed:
  (use "git rm --cached <file>..." to unstage)
        new file:   DEPLOY_FINETUNE_UBUNTU24.md
        new file:   DEPLOY_SQLCODER_UBUNTU24.md
        ...
```

### Step 5: Create First Commit

```bash
git commit -m "Initial commit: Oracle EBS NL2SQL framework with LLaMA and SQLCoder"
```

Or more detailed:

```bash
git commit -m "Initial commit: Core framework and documentation

- LLaMA-3.3-70B fine-tuning pipeline (custom_oracle_llama/)
- SQLCoder-70B secondary model (custom_oracle_sqlcoder/)
- Dataset preprocessing and validation
- Complete deployment guides for Ubuntu 24.04 + CUDA 12.8
- Dual-model strategy documentation"
```

---

## First Push

```bash
# Push to GitHub (first time)
git branch -M main
git push -u origin main
```

Breakdown:
- `git branch -M main`: Rename current branch to `main`
- `git push -u origin main`: Push to GitHub and set upstream

Expected output:
```
Enumerating objects: 45, done.
Counting objects: 100% (45/45), done.
Delta compression using up to 8 threads
Compressing objects: 100% (42/42), done.
Writing objects: 100% (45/45), 523.45 KiB, done.
...
* [new branch]      main -> main
Branch 'main' set up to track remote branch 'main' from 'origin'.
```

### Verify on GitHub

1. Go to https://github.com/YOUR-USERNAME/askGuru-SQL
2. Verify files appear
3. Check commit message

---

## Daily Workflow

### Making Changes

```bash
# 1. Make edits to files
# (edit DEPLOY_SQLCODER_UBUNTU24.md, add new features, etc.)

# 2. Check status
git status

# 3. Stage changes
git add DEPLOY_SQLCODER_UBUNTU24.md
git add custom_oracle_sqlcoder/

# 4. Commit with message
git commit -m "Update: SQLCoder deployment guide with CUDA 12.8 support"

# 5. Push to GitHub
git push
```

### Commit Message Best Practices

Good messages:
```
# Feature
git commit -m "Add: SQLCoder JOIN validator"

# Fix
git commit -m "Fix: Preprocessing function for SQLCoder prompts"

# Documentation
git commit -m "Docs: Update deployment guide with CUDA 12.8"

# Refactor
git commit -m "Refactor: Consolidate preprocessing utilities"
```

Bad messages:
```
git commit -m "update"
git commit -m "changes"
git commit -m "fix bug"
```

### Detailed Workflow (Real Example)

```bash
# 1. Work on new feature
# Edit: custom_oracle_sqlcoder/inference_oracle_sqlcoder.py

# 2. Check what changed
git diff custom_oracle_sqlcoder/inference_oracle_sqlcoder.py

# 3. Stage the file
git add custom_oracle_sqlcoder/inference_oracle_sqlcoder.py

# 4. Commit
git commit -m "Improve: Add batch inference support to SQLCoder"

# 5. Push
git push

# 6. Verify on GitHub
# https://github.com/YOUR-USERNAME/askGuru-SQL/commits
```

---

## Pulling Updates

### From Another Machine

```bash
# Clone repository (first time)
git clone https://github.com/Akellags/askGuru-SQL.git
cd askGuru-SQL

# Verify remote
git remote -v
```

### Get Latest Changes

```bash
# Fetch latest from GitHub (doesn't merge)
git fetch

# Pull latest and merge (recommended)
git pull origin main

# Or in one step (pulls from upstream if configured)
git pull
```

### Merging Changes

If you made local changes and want to pull:

```bash
# Pull with merge
git pull origin main

# Or stash local changes, then pull
git stash
git pull origin main
git stash pop
```

---

## Branching Strategy

### Creating a Feature Branch

```bash
# Create and switch to new branch
git checkout -b feature/sqlcoder-join-fixes

# Or in newer Git
git switch -c feature/sqlcoder-join-fixes

# Work on feature
# ... make changes ...

# Commit
git commit -m "Add: Improved JOIN validation for SQLCoder"

# Push feature branch
git push -u origin feature/sqlcoder-join-fixes
```

### Pull Request (Optional but Recommended)

1. Go to GitHub repository
2. Click "Pull requests" tab
3. Click "New pull request"
4. Select:
   - **Base**: `main`
   - **Compare**: `feature/sqlcoder-join-fixes`
5. Add title and description
6. Click "Create pull request"
7. Review, discuss, merge

### Merging Locally

```bash
# Switch to main
git checkout main

# Pull latest main
git pull origin main

# Merge feature branch
git merge feature/sqlcoder-join-fixes

# Delete feature branch (optional)
git branch -d feature/sqlcoder-join-fixes
git push origin --delete feature/sqlcoder-join-fixes

# Push merged changes
git push origin main
```

---

## Common Commands

### Status and History

```bash
# Current status
git status

# View recent commits
git log

# View last 5 commits
git log -5

# View commits by author
git log --author="Your Name"

# View formatted log
git log --oneline --graph --all
```

### Viewing Changes

```bash
# Diff of unstaged changes
git diff

# Diff of staged changes
git diff --staged

# Diff between branches
git diff main feature/new-feature

# Show specific commit
git show abc1234
```

### Undoing Changes

```bash
# Undo unstaged changes
git checkout -- filename.md

# Unstage file
git reset HEAD filename.md

# Undo last commit (keep changes)
git reset --soft HEAD~1

# Undo last commit (discard changes)
git reset --hard HEAD~1
```

### Tags and Releases

```bash
# Create tag
git tag -a v1.0.0 -m "Version 1.0.0: Initial release"

# Push tags
git push origin --tags

# List tags
git tag -l
```

---

## Recommended Directory Structure in GitHub

```
askGuru-SQL/
├── README.md                              ← Overview
├── GITHUB_WORKFLOW.md                     ← This file
├── DUAL_MODEL_STRATEGY.md
├── CUDA_12.8_UPDATE.md
├── ubuntu24.04LTS_R570_CUDA12.8.md
│
├── DEPLOY_FINETUNE_UBUNTU24.md           ← Training guides
├── DEPLOY_INFERENCE_UBUNTU24.md
├── DEPLOY_SQLCODER_UBUNTU24.md
├── DEPLOYMENT_GUIDE_SUMMARY.md
│
├── custom_oracle_llama/                  ← LLaMA model code
│   ├── __init__.py
│   ├── _preprocessing_utils.py
│   ├── sft_oracle_llama70b_lora.py
│   ├── inference/
│   └── README.md
│
├── custom_oracle_sqlcoder/               ← SQLCoder model code
│   ├── _preprocessing_sqlcoder.py
│   ├── sft_oracle_sqlcoder70b_lora.py
│   ├── inference_oracle_sqlcoder.py
│   ├── sqlcoder_join_validator.py
│   └── README.md
│
├── train/                                 ← Shared training framework
│   ├── trainer/
│   ├── model/
│   └── config/
│
├── data/                                  ← Dataset directory
│   ├── oracle_sft_conversations/
│   │   ├── oracle_sft_conversations_train.json
│   │   ├── oracle_sft_conversations_val.json
│   │   └── oracle_sft_conversations_test.json
│   └── oracle_sft_config.json
│
├── .gitignore                            ← Git ignore rules
└── requirements.txt                      ← Dependencies
```

---

## Best Practices for This Project

### What TO Commit

✅ Source code (`.py` files)  
✅ Configuration files (`.yaml`, `.json`)  
✅ Documentation (`.md` files)  
✅ Scripts (`.sh` files)  
✅ Requirements (`requirements.txt`)  

### What NOT to Commit

❌ Model weights (too large)  
❌ Training checkpoints (use `.gitignore`)  
❌ Large datasets  
❌ Virtual environments (`venv/`)  
❌ IDE settings (`.vscode/`, `.idea/`)  
❌ Logs (`logs/`, `*.log`)  
❌ Sensitive files (`.env`, API keys)  

### Commit Frequency

- **Small changes**: Commit after each logical change
- **Training runs**: Commit after code changes, not after every epoch
- **Documentation**: Commit documentation updates separately
- **Example**: 
  ```bash
  git commit -m "Add: SQLCoder preprocessing function"  # One commit
  git commit -m "Docs: Update README with usage examples"  # Another commit
  ```

---

## Troubleshooting

### "fatal: not a git repository"

```bash
# You're not in the project directory
cd /path/to/askGuru-SQL
git status
```

### "fatal: unable to access repository"

```bash
# GitHub credentials not set up
# Try HTTPS (will prompt for password) or SSH

# HTTPS with token (recommended)
git remote set-url origin https://TOKEN@github.com/USERNAME/askGuru-SQL.git

# Or SSH (requires key setup)
git remote set-url origin git@github.com:USERNAME/askGuru-SQL.git
```

### "Your branch is ahead of origin/main"

```bash
# You have local commits not pushed
git push origin main
```

### "conflict: merge conflict in ..."

```bash
# Pull conflicts occurred
# Edit the conflicted file:
# - Remove conflict markers (<<<, ===, >>>)
# - Keep desired version
# - Save

git add conflicted_file.py
git commit -m "Resolve: Merge conflict in preprocessing"
git push
```

### Large file errors

```bash
# Accidentally committed large file
git rm --cached large_model.bin
git commit -m "Remove: Large model file"
git push

# Add to .gitignore for future
echo "*.bin" >> .gitignore
git add .gitignore
git commit -m "Update: Add large files to gitignore"
```

---

## Next Steps

1. **Initialize**: Follow "Initial Setup" section above
2. **Push**: Complete "First Push"
3. **Verify**: Check GitHub repo
4. **Document**: Create issues/milestones for tasks
5. **Automate** (Optional): Set up GitHub Actions for testing

---

## References

- [Git Documentation](https://git-scm.com/doc)
- [GitHub Guides](https://guides.github.com/)
- [Pro Git Book](https://git-scm.com/book/en/v2)
- [Conventional Commits](https://www.conventionalcommits.org/)

---

## Quick Reference Card

```bash
# Clone
git clone https://github.com/user/askGuru-SQL.git

# First time setup
git init
git remote add origin https://github.com/user/askGuru-SQL.git
git add .
git commit -m "Initial commit"
git branch -M main
git push -u origin main

# Daily workflow
git status
git add filename.py
git commit -m "Feature: Description"
git push

# Get updates
git pull origin main

# Create feature branch
git checkout -b feature/name
git push -u origin feature/name

# Check differences
git diff
git log --oneline
```

