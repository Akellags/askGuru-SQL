# /llamaSFT Volume Migration Summary

**Complete migration from root/home directories to /llamaSFT mounted volume**

---

## Migration Completed ✅

All deployment scripts and guides have been updated to use:
- **Base Directory**: `/llamaSFT` (mounted volume)
- **HuggingFace Cache**: `/llamaSFT/hf_home`
- **User**: `ubuntu` (existing system user, no trainuser creation)
- **Data Storage**: `/llamaSFT/data/`, `/llamaSFT/models/`, `/llamaSFT/outputs/`

---

## Files Updated

### Documentation Files (4)

| File | Changes |
|------|---------|
| **DEPLOY_FINETUNE_UBUNTU24.md** | Removed trainuser creation, updated all paths to `/llamaSFT`, added HF_HOME setup |
| **DEPLOY_SQLCODER_UBUNTU24.md** | Updated paths to `/llamaSFT`, added HF_HOME setup, updated training commands |
| **DEPLOY_INFERENCE_UBUNTU24.md** | Updated paths to `/llamaSFT`, updated vLLM service paths |
| **VOLUME_SETUP.md** | **NEW** - Comprehensive guide for /llamaSFT initialization |

### Deployment Scripts (2)

| File | Changes |
|------|---------|
| **deploy_finetune.sh** | Updated all workspace variables to `/llamaSFT`, added `export HF_HOME` |
| **deploy_inference.sh** | Updated all workspace variables to `/llamaSFT`, added `export HF_HOME` |

---

## Key Changes Summary

### 1. Directory Structure Changes

#### Before
```
~/askguru-sql-training/
├── models/
├── data/
├── outputs/
└── venv/
```

#### After
```
/llamaSFT/
├── models/                    # Large model files
├── data/                      # Datasets
├── outputs/                   # Training outputs
├── logs/                      # Training logs
├── hf_home/                   # HuggingFace cache
├── checkpoints/               # Additional checkpoints
└── venv/                      # Python environment
```

### 2. User Changes

#### Before
- Created new user `trainuser`
- All operations as `trainuser`
- Home directory: `~` = `/home/trainuser/`

#### After
- Use existing `ubuntu` user
- No additional user creation
- Home directory: `/home/ubuntu/`
- No `sudo` needed for regular operations

### 3. Environment Variables

#### Added
```bash
# Set HuggingFace cache to volume
export HF_HOME=/llamaSFT/hf_home

# All model downloads go to /llamaSFT/hf_home
# Not ~/.cache/huggingface
```

### 4. Path Updates

All references updated:
```
~/askguru-sql-training  → /llamaSFT
~                       → /llamaSFT (for project files)
/home/trainuser         → /home/ubuntu
```

---

## Installation Quick Reference

### Setup (One-time)
```bash
# As ubuntu user, on /llamaSFT volume
cd /llamaSFT

# Create directories
mkdir -p models data outputs logs hf_home checkpoints

# Clone code
git clone https://github.com/YOUR-USERNAME/askGuru-SQL.git .

# Setup HF_HOME
export HF_HOME=/llamaSFT/hf_home
echo 'export HF_HOME=/llamaSFT/hf_home' >> ~/.bashrc

# Create venv
python3.12 -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch transformers datasets accelerate deepspeed peft
```

### Training LLaMA
```bash
cd /llamaSFT
source venv/bin/activate

accelerate launch --config_file train/config/zero3.yaml \
  custom_oracle_llama/sft_oracle_llama70b_lora.py \
  --model_name_or_path /llamaSFT/models/llama-3.3-70b-instruct \
  --data_path /llamaSFT/data/oracle_sft_conversations/oracle_sft_conversations_train.json \
  --output_dir /llamaSFT/outputs/oracle_llama70b_lora \
  ...
```

### Training SQLCoder
```bash
cd /llamaSFT
source venv/bin/activate

accelerate launch --config_file train/config/zero3.yaml \
  custom_oracle_sqlcoder/sft_oracle_sqlcoder70b_lora.py \
  --model_name_or_path defog/sqlcoder-70b-alpha \
  --data_path /llamaSFT/data/oracle_sft_conversations/oracle_sft_conversations_train.json \
  --output_dir /llamaSFT/outputs/oracle_sqlcoder70b_lora \
  ...
```

### Inference
```bash
cd /llamaSFT
source venv/bin/activate

python -m vllm.entrypoints.openai.api_server \
  --model /llamaSFT/models/oracle_llama70b_merged \
  --dtype bfloat16 \
  --port 8000
```

---

## Storage Planning

### Disk Space Requirements

| Component | Size | Location |
|-----------|------|----------|
| LLaMA-3.3-70B base | ~140GB | `/llamaSFT/models/` |
| SQLCoder-70B base | ~140GB | `/llamaSFT/models/` |
| Training checkpoints | ~20-50GB | `/llamaSFT/outputs/` |
| Merged models | ~280GB | `/llamaSFT/models/` |
| Quantized (4-bit) | ~70GB | `/llamaSFT/models/` |
| HF cache | ~5-20GB | `/llamaSFT/hf_home/` |
| Dataset + code | ~100MB | `/llamaSFT/data/` & root |
| **Total** | **~600-700GB** | *All on one volume* |

### Sequential Training (Recommended for 500GB volume)
```
Train LLaMA (10h)
  ├─ Delete base model to free 140GB
  ├─ Keep LoRA weights (~300MB)
  └─ Available: 360GB

Train SQLCoder (10h)
  ├─ Download base: 140GB
  ├─ Training: 10GB
  └─ Total: 500GB ✅ FIT

Merge both models
  └─ 360GB final artifacts
```

---

## Verification Checklist

### Before Training
```bash
# Verify volume mounted
mount | grep llamaSFT  # Should show /llamaSFT

# Verify ownership
ls -ld /llamaSFT  # Should be ubuntu:ubuntu

# Verify directories
ls -la /llamaSFT/  # Should show models/, data/, outputs/, etc.

# Verify HF_HOME
echo $HF_HOME  # Should print: /llamaSFT/hf_home

# Verify venv
source /llamaSFT/venv/bin/activate
python --version  # Should work

# Verify dataset
ls -lh /llamaSFT/data/oracle_sft_conversations/

# Verify available space
df -h /llamaSFT  # Should show ~100GB+ free
```

### After Training
```bash
# Check outputs
ls -lh /llamaSFT/outputs/oracle_llama70b_lora/adapter_model.bin

# Check models
du -sh /llamaSFT/models/*

# Check space usage
du -sh /llamaSFT
df -h /llamaSFT

# Verify HF cache populated
ls /llamaSFT/hf_home/hub/
```

---

## Important Notes

### ⚠️ Do Not

- ❌ Do NOT use `sudo` for regular operations
- ❌ Do NOT create `trainuser` - use existing `ubuntu` user
- ❌ Do NOT set HF_HOME to `~/.cache/huggingface` - use `/llamaSFT/hf_home`
- ❌ Do NOT store models outside `/llamaSFT` - volume will run out of space

### ✅ Always

- ✅ Run all commands as `ubuntu` user
- ✅ Run from `/llamaSFT` directory
- ✅ Verify `/llamaSFT` is mounted before starting
- ✅ Set `HF_HOME=/llamaSFT/hf_home` before training
- ✅ Use full paths like `/llamaSFT/models/` in scripts
- ✅ Monitor disk space during training (`df -h /llamaSFT`)

---

## Reference Documentation

- **VOLUME_SETUP.md** - Complete volume initialization guide
- **DEPLOY_FINETUNE_UBUNTU24.md** - LLaMA training deployment
- **DEPLOY_SQLCODER_UBUNTU24.md** - SQLCoder training deployment
- **DEPLOY_INFERENCE_UBUNTU24.md** - vLLM inference deployment
- **GITHUB_WORKFLOW.md** - Git/GitHub operations
- **CUDA_12.8_UPDATE.md** - CUDA 12.8 compatibility

---

## Rollback (If Needed)

If you need to revert to home directory approach:

```bash
# 1. Stop all training processes
pkill -f "accelerate launch"
pkill -f "vllm"

# 2. Copy data from /llamaSFT to home
cp -r /llamaSFT/data ~/
cp -r /llamaSFT/models ~/
cp -r /llamaSFT/outputs ~/

# 3. Update scripts to use home paths again
# Edit deploy_finetune.sh, deploy_inference.sh
# Change:
#   WORKSPACE="/llamaSFT" 
# Back to:
#   WORKSPACE="${HOME}/askguru-sql-training"

# 4. Update .md files to use home paths
```

---

## Summary

| Aspect | Before | After |
|--------|--------|-------|
| Base Directory | `~/askguru-sql-training` | `/llamaSFT` |
| User | `trainuser` (created) | `ubuntu` (existing) |
| HF Cache | `~/.cache/huggingface` | `/llamaSFT/hf_home` |
| Storage | Root disk (risky) | Mounted volume (safe) |
| Paths | Relative `~/` | Absolute `/llamaSFT/` |
| Scripts | Uses PROJECT_ROOT | Uses fixed `/llamaSFT` |

**All changes complete and ready for production training!**

