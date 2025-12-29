# Deployment Guide Summary

Complete deployment scripts and guides for Oracle EBS NL2SQL on Ubuntu 24

---

## Overview

This package includes **complete manual and automated deployment solutions** for:

1. **Fine-tuning** LLaMA-3.1-70B on 8×A100-80GB GPUs
2. **Inference** with vLLM on 1×A100-80GB GPU

---

## Files Included

### Fine-tuning (8×A100)

#### Manual Guide: `DEPLOY_FINETUNE_UBUNTU24.md`
- **Purpose:** Step-by-step manual deployment instructions
- **Audience:** Operators who prefer to understand each step
- **Content:**
  - System prerequisites and hardware requirements
  - NVIDIA driver & CUDA installation
  - Python environment setup
  - PyTorch installation with CUDA 12.1
  - LLaMA model download from Hugging Face
  - Training data preparation and validation
  - DeepSpeed ZeRO-3 configuration
  - Training launch with Accelerate
  - Monitoring and troubleshooting
  - ~500 lines of detailed instructions
- **Time to complete:** ~3-4 hours setup + 4-6 hours training

#### Automated Script: `deploy_finetune.sh`
- **Purpose:** One-command automated fine-tuning deployment
- **Audience:** Operators who want quick, reliable deployment
- **Features:**
  - Full automation of all setup steps
  - Dependency installation
  - GPU verification
  - Model download with HF token handling
  - Dataset preparation and validation
  - DeepSpeed config generation
  - Training with automatic logging
  - Progress monitoring instructions
  - Resume from checkpoint support
  - ~600 lines of production-grade bash
- **Usage:** 
  ```bash
  chmod +x deploy_finetune.sh
  ./deploy_finetune.sh
  # or with options:
  ./deploy_finetune.sh --skip-deps --skip-model
  ```
- **Time to complete:** ~45 minutes setup + 4-6 hours training

### Inference (1×A100)

#### Manual Guide: `DEPLOY_INFERENCE_UBUNTU24.md`
- **Purpose:** Step-by-step inference deployment instructions
- **Audience:** Operators who prefer manual control and understanding
- **Content:**
  - System prerequisites
  - NVIDIA driver & CUDA installation
  - PyTorch + vLLM dependencies
  - Model merge (LoRA + base)
  - 4-bit quantization (AWQ)
  - vLLM server setup (multiple methods)
  - Client integration (curl, Python, OpenAI SDK)
  - Health checks and testing
  - Production monitoring with GPU/metrics
  - Performance benchmarking
  - Troubleshooting (OOM, slow inference, crashes)
  - ~450 lines of detailed instructions
- **Time to complete:** ~2-3 hours setup + 30 minutes merge/quantize

#### Automated Script: `deploy_inference.sh`
- **Purpose:** One-command automated inference deployment
- **Audience:** Operators who want quick, ready-to-serve deployment
- **Features:**
  - Full automation of all setup steps
  - GPU verification
  - Dependency installation
  - Model verification
  - Automatic merge & quantization
  - vLLM server startup
  - Health checks
  - Inference testing (optional)
  - systemd service creation
  - Production monitoring guide
  - ~600 lines of production-grade bash
- **Usage:**
  ```bash
  chmod +x deploy_inference.sh
  ./deploy_inference.sh
  # or with options:
  ./deploy_inference.sh --skip-deps --skip-merge --test
  ```
- **Time to complete:** ~2-3 hours total

---

## Quick Start Guide

### Option A: Fully Automated (Recommended for CI/CD)

#### Fine-tuning:
```bash
# On 8×A100 training machine
chmod +x deploy_finetune.sh
./deploy_finetune.sh --skip-deps  # if deps already installed

# Monitor progress
tail -f training_workspace/logs/training_*.log
watch -n 1 nvidia-smi
```

#### Inference:
```bash
# On 1×A100 inference machine (after training)
# First, copy fine-tuned LoRA from training machine:
rsync -avz user@training:/path/to/outputs/oracle_llama70b_lora ./models/

# Then deploy inference server
chmod +x deploy_inference.sh
./deploy_inference.sh --test

# Monitor
curl http://localhost:8000/v1/models
watch -n 1 nvidia-smi
```

### Option B: Manual Step-by-Step (Recommended for Learning)

#### Fine-tuning:
```bash
# Read the manual guide
cat DEPLOY_FINETUNE_UBUNTU24.md

# Follow each section step by step
# 1. Check hardware prerequisites
# 2. Install system packages
# 3. Setup NVIDIA drivers
# 4. Create Python environment
# 5. Install PyTorch
# 6. Download model
# 7. Prepare dataset
# 8. Run training

# Start training
accelerate launch --config_file train/config/zero3_a100.yaml \
  custom_oracle_llama/sft_oracle_llama70b_lora.py \
  --model_name_or_path models/llama-3.1-70b-instruct \
  --data_path data/oracle_sft_conversations/oracle_sft_conversations_train.json \
  --output_dir outputs/oracle_llama70b_lora
```

#### Inference:
```bash
# Read the manual guide
cat DEPLOY_INFERENCE_UBUNTU24.md

# Follow each section:
# 1. Check hardware
# 2. Install system packages
# 3. Setup NVIDIA drivers
# 4. Create Python environment
# 5. Install vLLM
# 6. Merge & quantize model
# 7. Start vLLM server
# 8. Test with client

# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
  --model ./models/merged_oracle_llama70b_awq4 \
  --max-num-seqs 4 \
  --port 8000
```

---

## Architecture Overview

```
Fine-tuning (8×A100)
├─ Input: 4,822 training examples in XiYan format
│  └─ 3,857 train / 482 val / 483 test
├─ Training: DeepSpeed ZeRO-3 + LoRA BF16
│  └─ 3 epochs, 4 batch size per GPU, 8 GPUs = 32 total batch
├─ Output: LoRA adapter (~30MB)
│  └─ outputs/oracle_llama70b_lora/
└─ Duration: 4-6 hours

Inference (1×A100)
├─ Input: Base model + LoRA adapter
├─ Merge: Combine LoRA weights into base model
├─ Quantize: 4-bit AWQ quantization (~20GB → 35GB model size)
├─ Serve: vLLM OpenAI-compatible API
│  ├─ Max 4 concurrent sequences
│  ├─ Max 512 tokens per response
│  ├─ Temperature 0.0 (deterministic)
│  └─ 92% GPU memory utilization
└─ API: OpenAI-compatible (curl, Python, any SDK)
```

---

## Configuration Parameters

### Fine-tuning Parameters
| Parameter | Value | Notes |
|-----------|-------|-------|
| **Model** | LLaMA-3.1-70B-Instruct | Base foundation model |
| **Hardware** | 8×A100-80GB | Recommended for production |
| **Training Method** | LoRA BF16 | Lower memory than full FT |
| **Batch Size** | 4 per GPU (32 total) | Adjust for OOM |
| **Learning Rate** | 2.0e-4 | Standard for LoRA |
| **Warmup Steps** | 500 | Linear warmup |
| **Num Epochs** | 3 | Can be adjusted |
| **LoRA Rank (r)** | 32 | Balance quality vs size |
| **LoRA Alpha** | 32 | Usually same as r |
| **Max Sequence Length** | 8192 | Oracle SQL context length |
| **Gradient Checkpointing** | True | Saves VRAM |
| **Optimizer** | AdamW 8-bit | Memory efficient |
| **Framework** | Accelerate + DeepSpeed | Handles ZeRO-3 |

### Inference Parameters
| Parameter | Value | Notes |
|-----------|-------|-------|
| **Model** | Quantized 4-bit AWQ | ~35GB after quantization |
| **Hardware** | 1×A100-80GB | Sufficient for 4 concurrent |
| **GPU Memory** | 92% utilization | Near-optimal usage |
| **Max Sequences** | 4 | Concurrent requests |
| **Max Output Tokens** | 512 | SQL queries typically <300 |
| **Temperature** | 0.0 | Deterministic (no randomness) |
| **Max Model Length** | 8192 | Matches training context |
| **Framework** | vLLM | Optimized inference engine |
| **API** | OpenAI-compatible | Standard /v1/completions |

---

## Pre-deployment Checklist

### Fine-tuning Machine

- [ ] Ubuntu 24.04 LTS installed
- [ ] 8× NVIDIA A100-80GB GPUs (or 8 compatible GPUs)
- [ ] NVIDIA drivers installed (latest)
- [ ] CUDA 12.1+ installed
- [ ] cuDNN installed
- [ ] 1TB+ disk space available
- [ ] High-bandwidth internet (for model download)
- [ ] Hugging Face account and API token
- [ ] Project repository cloned
- [ ] Training data at `data/oracle_sft_conversations/`

### Inference Machine

- [ ] Ubuntu 24.04 LTS installed
- [ ] 1× NVIDIA A100-80GB GPU
- [ ] NVIDIA drivers installed (latest)
- [ ] CUDA 12.1+ installed
- [ ] 200GB+ disk space available
- [ ] Port 8000 available (or use --port)
- [ ] Project repository cloned
- [ ] LoRA adapter files available (from training)

---

## Troubleshooting Quick Reference

### Fine-tuning Issues

**GPU Out of Memory**
```bash
# Reduce batch size
--per_device_train_batch_size 2

# Or reduce sequence length
--model_max_length 4096

# Or enable more gradient checkpointing
--gradient_checkpointing true
```

**Training stuck/slow**
```bash
# Check GPU usage
nvidia-smi

# Check process
ps aux | grep accelerate

# Resume from checkpoint
./deploy_finetune.sh --resume-from checkpoint-500
```

**Model download fails**
```bash
# Re-authenticate with Hugging Face
huggingface-cli logout
huggingface-cli login
```

### Inference Issues

**Server won't start**
```bash
# Check port available
lsof -i :8000

# Try different port
./deploy_inference.sh --port 8001

# Check GPU
nvidia-smi

# Check vLLM logs
tail -f inference_workspace/logs/vllm_*.log
```

**Slow inference**
```bash
# Reduce concurrent sequences
--max-num-seqs 2

# Check GPU temperature
nvidia-smi -q -d TEMPERATURE

# Verify quantization loaded
ps aux | grep vllm  # should show awq in command
```

**Poor SQL quality**
```bash
# Verify fine-tuned model loaded
curl http://localhost:8000/v1/models

# Check prompt format matches training
# Should use [User Question] (English), not 【用户问题】(Chinese)
```

---

## Performance Expectations

### Fine-tuning Performance

**With 8×A100-80GB NVLink:**
- Data loading: ~5 minutes
- Per epoch: ~1.5 hours
- Total (3 epochs): ~4-5 hours
- Checkpoint frequency: Every 500 steps (~30 min)

**GPU Memory Usage:**
- Per GPU: ~65-70GB (with ZeRO-3 offloading)
- Activation memory: 8192 seq length × 4 batch/GPU
- LoRA weights: ~30MB per adapter

### Inference Performance

**On 1×A100-80GB with 4 concurrent:**
- Model loading: ~2 minutes
- Cold start latency: 5-10 seconds (first token)
- Throughput: 40-80 tokens/second (depending on SQL complexity)
- Memory usage: ~35GB (quantized 4-bit model)
- Concurrent users: 4 optimal, can go to 8 with latency increase

### Estimated Costs (AWS pricing, approximate)

**Fine-tuning on p5.48xlarge (8×A100):**
- Hardware: $40/hour × 5 hours = $200
- Storage: $0.023/GB-month × 200GB = $5

**Inference on g5.2xlarge (1×A100):**
- Hardware: $2.48/hour (reserved) to $4.48/hour (on-demand)
- Storage: ~$5/month

---

## Next Steps After Deployment

### After Fine-tuning:
1. Transfer LoRA adapter to inference machine
2. Run evaluation on test set
3. Deploy inference server
4. Integration test with your application

### After Inference Deployment:
1. Load test with 4+ concurrent users
2. Monitor GPU memory and latency
3. Setup health checks and auto-restart
4. Configure logging and metrics
5. Train 2nd version (if needed) with more data

---

## Support & References

### Documentation Files
- `DEPLOY_FINETUNE_UBUNTU24.md` - Fine-tuning manual
- `DEPLOY_INFERENCE_UBUNTU24.md` - Inference manual
- `custom_oracle_llama/README.md` - Model-specific docs
- `DATASET_BUILD_SUMMARY.md` - Dataset preparation
- `DATA_PIPELINE_PLAN.md` - Pipeline architecture

### External References
- **LLaMA Model:** https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct
- **vLLM:** https://github.com/vllm-project/vllm
- **DeepSpeed:** https://github.com/microsoft/DeepSpeed
- **PEFT/LoRA:** https://github.com/huggingface/peft
- **AWQ Quantization:** https://github.com/mit-han-lab/awq

---

## Version Information

- **Created:** 2025-12-29
- **Ubuntu Version:** Ubuntu 24.04 LTS
- **PyTorch:** 2.3.1
- **CUDA:** 12.1
- **Transformers:** 4.42.3
- **vLLM:** 0.4.3
- **DeepSpeed:** 0.12.0
- **LoRA/PEFT:** 0.11.1

---

**Last Updated:** 2025-12-29

