# /llamaSFT Volume Setup Guide

**Complete initialization guide for the /llamaSFT mounted volume**

---

## Overview

All training and inference for askGuru-SQL happens on the `/llamaSFT` mounted volume:
- **Base Directory**: `/llamaSFT`
- **HuggingFace Cache**: `/llamaSFT/hf_home`
- **User**: `ubuntu`
- **Primary Purpose**: Store code, models, datasets, and training outputs

---

## Prerequisites

### 1. Verify Volume is Mounted

```bash
# Check if /llamaSFT exists and is mounted
mount | grep llamaSFT

# Should show something like:
# /dev/nvmeXnX on /llamaSFT type ext4 (rw,relatime)

# If not mounted, contact infrastructure team
```

### 2. Verify Ownership

```bash
# Check permissions and owner
ls -ld /llamaSFT

# Should show:
# drwxr-xr-x ... ubuntu ubuntu ... /llamaSFT
```

### 3. Verify Ubuntu User

```bash
# Verify you're running as ubuntu user
whoami  # Should output: ubuntu

# Check ubuntu user groups
id ubuntu  # Should include docker, sudo, etc.
```

---

## Initial Setup (First Time)

### 1. Create Directory Structure

```bash
cd /llamaSFT

# Create subdirectories
mkdir -p models                       # Large model files
mkdir -p data                         # Datasets
mkdir -p outputs                      # Training checkpoints & outputs
mkdir -p logs                         # Training logs
mkdir -p hf_home                      # HuggingFace cache
mkdir -p checkpoints                  # Additional checkpoints

# Verify
ls -la /llamaSFT/
```

### 2. Set HuggingFace Cache (One-time)

```bash
# Create ~/.bashrc if doesn't exist
touch /home/ubuntu/.bashrc

# Add HF_HOME to environment
export HF_HOME=/llamaSFT/hf_home
echo 'export HF_HOME=/llamaSFT/hf_home' >> /home/ubuntu/.bashrc

# Verify
source /home/ubuntu/.bashrc
echo $HF_HOME  # Should print: /llamaSFT/hf_home
```

### 3. Clone Project from GitHub

```bash
cd /llamaSFT

# Clone the repository (replace YOUR-USERNAME)
git clone https://github.com/YOUR-USERNAME/askGuru-SQL.git .

# Or copy existing files
# cp -r /source/path/* .

# Verify
ls -la /llamaSFT/
# Should show: custom_oracle_llama/, custom_oracle_sqlcoder/, data/, deploy_*.sh, etc.
```

### 4. Setup Python Virtual Environment

```bash
cd /llamaSFT

# Create virtual environment
python3.12 -m venv venv

# Activate it
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install dependencies (see DEPLOY_FINETUNE_UBUNTU24.md or DEPLOY_SQLCODER_UBUNTU24.md)
pip install torch transformers datasets accelerate deepspeed peft
```

### 5. Verify Setup

```bash
# Check all directories exist
ls -la /llamaSFT/ | grep -E "^d"

# Check venv is created
ls -la /llamaSFT/venv/bin/python*

# Check git repo is initialized
git -C /llamaSFT status

# Test Python environment
python /llamaSFT/venv/bin/python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

---

## Directory Structure

```
/llamaSFT/
├── venv/                                    # Python virtual environment
│   ├── bin/python3
│   ├── lib/python3.12/site-packages/
│   └── ...
│
├── custom_oracle_llama/                    # LLaMA-3.1-70B training code
│   ├── _preprocessing_utils.py
│   ├── sft_oracle_llama70b_lora.py
│   └── ...
│
├── custom_oracle_sqlcoder/                 # SQLCoder-70B training code
│   ├── _preprocessing_sqlcoder.py
│   ├── sft_oracle_sqlcoder70b_lora.py
│   └── ...
│
├── train/                                  # Shared training framework
│   ├── trainer/
│   ├── model/
│   └── config/
│       ├── zero3.yaml
│       └── dp_zero3.json
│
├── data/                                   # Dataset directory
│   ├── oracle_sft_conversations/
│   │   ├── oracle_sft_conversations_train.json
│   │   ├── oracle_sft_conversations_val.json
│   │   └── oracle_sft_conversations_test.json
│   └── oracle_sft_config.json
│
├── models/                                 # Large model files (>100GB)
│   ├── llama-3.1-70b-instruct/           # Downloaded base model
│   ├── oracle_llama70b_lora/             # LoRA weights
│   ├── oracle_llama70b_merged/           # Merged model
│   ├── oracle_llama70b_4bit/             # Quantized model
│   ├── oracle_sqlcoder70b_lora/          # SQLCoder LoRA
│   └── ...
│
├── outputs/                                # Training outputs
│   ├── oracle_llama70b_lora/             # LLaMA training outputs
│   │   ├── checkpoint-500/
│   │   ├── checkpoint-1000/
│   │   └── adapter_model.bin              # Final LoRA weights
│   ├── oracle_sqlcoder70b_lora/          # SQLCoder training outputs
│   └── ...
│
├── logs/                                   # Training logs
│   ├── training.log                      # Main training log
│   ├── tensorboard/
│   └── ...
│
├── hf_home/                               # HuggingFace cache
│   ├── hub/
│   └── ...
│
├── checkpoints/                           # Additional checkpoints
│
├── .git/                                   # Git repository metadata
├── .gitignore                             # Git ignore rules
├── README.md                              # Project README
├── GITHUB_WORKFLOW.md                     # Git/GitHub guide
├── DEPLOY_FINETUNE_UBUNTU24.md           # LLaMA deployment
├── DEPLOY_SQLCODER_UBUNTU24.md           # SQLCoder deployment
├── DEPLOY_INFERENCE_UBUNTU24.md          # Inference deployment
└── ...
```

---

## Common Operations

### Activate Virtual Environment

```bash
cd /llamaSFT
source venv/bin/activate

# Verify
python --version
which python  # Should show /llamaSFT/venv/bin/python
```

### Check Disk Space

```bash
# Total usage
du -sh /llamaSFT

# By directory
du -sh /llamaSFT/*

# Available space
df -h /llamaSFT
```

### Clean Cache (if running low on space)

```bash
# Clear HuggingFace cache (⚠️ Requires re-download)
rm -rf /llamaSFT/hf_home/hub

# Clear old checkpoints (keep final models!)
rm -rf /llamaSFT/outputs/*/checkpoint-* 

# Clear logs (after training complete)
rm -rf /llamaSFT/logs/*

# View space freed
du -sh /llamaSFT
```

### List All Models

```bash
ls -lh /llamaSFT/models/

# Show sizes
du -sh /llamaSFT/models/*

# Find largest files
find /llamaSFT/models -type f -exec ls -lh {} \; | sort -k5 -h
```

### Git Operations

```bash
cd /llamaSFT

# Check status
git status

# Pull latest changes
git pull origin main

# Push local changes
git add .
git commit -m "Update: Training results"
git push origin main

# View commit history
git log --oneline -10
```

---

## Storage Breakdown (Typical)

| Directory | Size | Notes |
|-----------|------|-------|
| `/llamaSFT/models/` | ~280-500GB | Base models + merged variants |
| `/llamaSFT/outputs/` | ~20-50GB | Checkpoints + final weights |
| `/llamaSFT/hf_home/` | ~5-20GB | HuggingFace cache |
| `/llamaSFT/data/` | ~100MB | JSON datasets (small) |
| `/llamaSFT/venv/` | ~2-5GB | Python packages |
| `/llamaSFT/logs/` | ~1-5GB | Training logs (can be cleared) |
| **Total** | **~310-560GB** | Varies by models trained |

---

## Troubleshooting

### "Permission denied" errors

```bash
# Check ownership
ls -ld /llamaSFT

# If not owned by ubuntu:ubuntu, contact infrastructure
# Don't use sudo for regular operations
```

### "No space left on device"

```bash
# Check available space
df -h /llamaSFT

# Free up space:
rm -rf /llamaSFT/logs/*           # Safe to delete
rm -rf /llamaSFT/hf_home/hub     # Safe but requires re-download
rm -rf /llamaSFT/outputs/*/checkpoint-*  # Keep final models!

# Show disk usage
du -sh /llamaSFT/*
```

### Python venv not activating

```bash
# Recreate venv
rm -rf /llamaSFT/venv
python3.12 -m venv /llamaSFT/venv

# Activate
source /llamaSFT/venv/bin/activate

# Reinstall packages
pip install torch transformers ...
```

### Git cannot commit (large files)

```bash
# Check if large files accidentally added
git lfs install  # Install Git LFS if needed

# Or remove from git
git rm --cached /llamaSFT/models/*
echo "models/" >> .gitignore
git add .gitignore
git commit -m "Remove large model files from git"
```

### HF_HOME not set

```bash
# Verify it's in bashrc
grep "HF_HOME" /home/ubuntu/.bashrc

# If missing, add it:
echo 'export HF_HOME=/llamaSFT/hf_home' >> /home/ubuntu/.bashrc
source /home/ubuntu/.bashrc

# Verify
echo $HF_HOME
```

---

## Maintenance

### Weekly Tasks

```bash
# Check disk space
df -h /llamaSFT

# Check git status
cd /llamaSFT && git status

# View recent logs
tail -20 /llamaSFT/logs/training.log
```

### Monthly Tasks

```bash
# Clean old logs (keep recent ones!)
find /llamaSFT/logs -name "*.log" -mtime +30 -delete

# Archive large logs
tar -czf /llamaSFT/logs/archive_$(date +%Y%m%d).tar.gz /llamaSFT/logs/*.log.old

# Pull latest code
cd /llamaSFT && git pull origin main
```

### Before Long Training

```bash
# Check available space
df -h /llamaSFT

# Should have at least 100GB free for checkpoints

# Clear unnecessary files
rm -rf /llamaSFT/hf_home/hub/*
rm /llamaSFT/logs/*.old

# Update code
cd /llamaSFT && git pull origin main
```

---

## Quick Start Checklist

- [ ] Volume mounted at `/llamaSFT`
- [ ] Owned by `ubuntu` user
- [ ] Directories created (models/, data/, outputs/, logs/, hf_home/)
- [ ] Code cloned from GitHub
- [ ] Virtual environment created and activated
- [ ] PyTorch installed and verified
- [ ] HF_HOME set to `/llamaSFT/hf_home`
- [ ] Dataset files present in `/llamaSFT/data/`
- [ ] Ready to start training!

```bash
# Final verification
cd /llamaSFT && source venv/bin/activate
python -c "import torch; print(f'Ready! PyTorch {torch.__version__}'); print(f'HF_HOME: {os.environ.get(\"HF_HOME\")}')"
```

---

## Support

For issues:
1. Check this guide for troubleshooting
2. Review deployment guides (DEPLOY_*.md)
3. Check training logs in `/llamaSFT/logs/`
4. Contact infrastructure team for volume issues

