# Oracle EBS NL2SQL Fine-tuning on 8×A100-80GB (Ubuntu 24)

**Complete manual deployment guide for LLaMA-3.3-70B LoRA fine-tuning**

---

## Table of Contents

1. [Prerequisites & Hardware](#prerequisites--hardware)
2. [Environment Setup](#environment-setup)
3. [NVIDIA Driver & CUDA Installation](#nvidia-driver--cuda-installation)
4. [PyTorch & Dependencies](#pytorch--dependencies)
5. [Model Download](#model-download)
6. [Training Data Preparation](#training-data-preparation)
7. [Training Configuration](#training-configuration)
8. [Running Fine-tuning](#running-fine-tuning)
9. [Monitoring & Troubleshooting](#monitoring--troubleshooting)
10. [Output & Post-Training](#output--post-training)

---

## Prerequisites & Hardware

### Hardware Requirements
- **GPU:** 8× NVIDIA A100-80GB (preferably NVLink interconnected)
- **CPU:** 128+ cores recommended
- **RAM:** 1TB+ system RAM
- **Storage:** 
  - 150GB for base model (LLaMA-3.3-70B)
  - 200GB for training data (4,822 examples)
  - 400GB for LoRA weights + checkpoints

### Network Requirements
- High-bandwidth internet (for HuggingFace model download)
- NVLink or 8xNVLink-C bandwidth between GPUs recommended for ZeRO-3

### Ubuntu 24 OS Requirements
- Fresh Ubuntu 24.04 LTS installation recommended
- Root or sudo access
- Kernel 6.5+ (default on Ubuntu 24)

---

## Environment Setup

### 1. Update System Packages

```bash
sudo apt-get update
sudo apt-get upgrade -y

# Install essential build tools
sudo apt-get install -y \
  build-essential \
  python3.12 \
  python3.12-dev \
  python3.12-venv \
  python3-pip \
  git \
  wget \
  curl \
  htop \
  nvtop \
  tmux \
  screen \
  openssh-server
```

### 2. Use Ubuntu User & /llamaSFT Volume

```bash
# Verify you're running as ubuntu user
whoami  # Should output: ubuntu

# Verify /llamaSFT is mounted and accessible
mount | grep llamaSFT
ls -ld /llamaSFT
```

### 3. Setup /llamaSFT Volume & Project

```bash
cd /llamaSFT

# Clone project from GitHub
git clone https://github.com/YOUR-USERNAME/askGuru-SQL.git . 2>/dev/null || echo "Existing repo or copy files manually"

# Create subdirectories
mkdir -p models outputs logs checkpoints

# Set HuggingFace cache to /llamaSFT/hf_home
mkdir -p /llamaSFT/hf_home
export HF_HOME=/llamaSFT/hf_home
echo 'export HF_HOME=/llamaSFT/hf_home' >> ~/.bashrc
source ~/.bashrc
```

### 4. Create Python Virtual Environment

```bash
cd /llamaSFT/askGuru-SQL

# Create virtual environment using Python 3.12
python3.12 -m venv .venv
source .venv/bin/activate

# Upgrade pip, setuptools, wheel
pip install --upgrade pip setuptools wheel
```

---

## NVIDIA Driver & CUDA Installation

### 1. Check Current GPU Status

```bash
# List GPUs
lspci | grep -i nvidia

# Check if nvidia-smi is already installed
nvidia-smi --version || echo "NVIDIA drivers not installed"
```

### 2. Install NVIDIA Drivers (Latest)

```bash
# Add NVIDIA repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID | sed -e 's/\.//g')
wget https://developer.download.nvidia.com/compute/cuda/repos/$distribution/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb

# Update and install CUDA toolkit
sudo apt-get update
sudo apt-get install -y cuda-toolkit libaio-dev

# Install cuDNN (required for training)
sudo apt-get install -y libcudnn8 libcudnn8-dev

# Add CUDA to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> /home/ubuntu/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> /home/ubuntu/.bashrc
source /home/ubuntu/.bashrc

# Verify installation
# Note: If you see "Failed to initialize NVML: Driver/library version mismatch", reboot the system: sudo reboot
nvidia-smi
nvcc --version
```

### 3. Verify CUDA & cuDNN

```bash
# Check CUDA availability
python3 << 'EOF'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_capability(i)})")
EOF
```

---

## PyTorch & Dependencies

### 1. Install PyTorch with CUDA 12.8 Support

```bash
# Ensure venv is activated
source /llamaSFT/askGuru-SQL/.venv/bin/activate

# Install PyTorch 2.5+ with CUDA 12.4 wheels (backward compatible with CUDA 12.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Verify PyTorch + CUDA
python << 'EOF'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
print(f"GPUs: {torch.cuda.device_count()}")
assert torch.cuda.is_available()
EOF
```

### 2. Install Core Dependencies

It is highly recommended to use the provided `req.txt` for fine-tuning as it contains the correct versions and optimized Flash-Attention wheels.

```bash
# Recommended: Install using the PyTorch extra index to resolve +cu124 versions
pip install -r req.txt --extra-index-url https://download.pytorch.org/whl/cu124

# Alternate Approach (Install PyTorch separately first):
# pip install torch==2.4.1+cu124 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
# pip install -r req.txt
```

### 3. Flash-Attention (Included in req.txt)

If not using `req.txt`, you must download and install the correct Flash-Attention wheel:
[Flash-Attention Releases](https://github.com/Dao-AILab/flash-attention/releases)

```bash
pip install flash_attn-2.8.3+cu12torch2.4cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
```

### 3. Install Optional Monitoring & Logging

```bash
# SwanLab for training visualization
pip install swanlab==0.6.0

# Weights & Biases (alternative to SwanLab)
pip install wandb

# TensorBoard
pip install tensorboard
```

### 4. Install Quantization Libraries (for post-training) -  not needed when training

```bash
# AWQ for quantization
pip install autoawq

# GPTQ alternative

```

### 5. Verify All Dependencies

```bash
python << 'EOF'
import torch
import transformers
import datasets
import accelerate
import deepspeed
import peft
import flash_attn

print("✓ All core dependencies installed successfully")
print(f"  PyTorch: {torch.__version__}")
print(f"  Transformers: {transformers.__version__}")
print(f"  Datasets: {datasets.__version__}")
print(f"  Accelerate: {accelerate.__version__}")
print(f"  DeepSpeed: {deepspeed.__version__}")
print(f"  PEFT: {peft.__version__}")
print(f"  Flash-Attention: {flash_attn.__version__}")
EOF
```

---

## Model Download

### 1. Download LLaMA-3.3-70B-Instruct

```bash
# Install git-lfs for large file handling
sudo apt-get install -y git-lfs

# Configure HuggingFace CLI
pip install huggingface-hub

# Login to HuggingFace (requires token from huggingface.co)
huggingface-cli login
# Paste your token when prompted

# Download model
mkdir -p /llamaSFT/models
cd /llamaSFT/models

huggingface-cli download meta-llama/Llama-3.3-70B-Instruct \
  --local-dir ./llama-3.3-70b-instruct \
  --token <add your token here>

```

### 2. Verify Model Download

```bash
# Check model size and files
ls -lh llama-3.3-70b-instruct/

# Verify tokenizer
python3 << 'EOF'
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("./llama-3.3-70b-instruct")
print(f"✓ Model loaded successfully")
print(f"  Vocab size: {len(tokenizer)}")
print(f"  BOS token: {tokenizer.bos_token}")
print(f"  EOS token: {tokenizer.eos_token}")
EOF
```

---

## Training Data Preparation

### 1. Verify Dataset Files

```bash
cd /llamaSFT/askGuru-SQL

# List available datasets
ls -lh data/oracle_sft_conversations/

# Expected files:
# - oracle_sft_conversations_train.json (3,857 examples)
# - oracle_sft_conversations_val.json (482 examples)
# - oracle_sft_conversations_test.json (483 examples)
```

### 2. Validate Dataset Format

```bash
source /llamaSFT/askGuru-SQL/.venv/bin/activate

python3 << 'EOF'
import json
import os

data_dir = "/llamaSFT/askGuru-SQL/data/oracle_sft_conversations"

for split in ["train", "val"]:
    path = f"{data_dir}/oracle_sft_conversations_{split}.json"
    if not os.path.exists(path):
        print(f"✗ Missing: {path}")
        continue
    
    with open(path, 'r') as f:
        data = json.load(f)
    
    print(f"\n✓ {split.upper()} dataset:")
    print(f"  Total examples: {len(data)}")
    
    # Check first example
    first = data[0]
    print(f"  Fields: {list(first.keys())}")
    print(f"  Conversations: {len(first.get('conversations', []))} turns")
    print(f"  SQL type: {first.get('sql_type', 'N/A')}")
    
    # Sample content
    user_prompt = first['conversations'][0]['content']
    print(f"  Prompt length: {len(user_prompt)} chars")
    print(f"  Has [User Question]: {'[User Question]' in user_prompt}")
EOF
```

### 3. Verify Directory Structure

```bash
# All files are in /llamaSFT already
ls -la /llamaSFT/
# Should show: models/, data/, outputs/, logs/, venv/, etc.
```

## Training Configuration

### 1. Review/Create DeepSpeed ZeRO-3 Config

```bash
# Check existing config
cat train/config/dp_zero3.json - (make usrer to check its there)

# If not present, create it:
cat > train/config/dp_zero3.json << 'EOF'
{
  "train_batch_size": 32,
  "train_micro_batch_size_per_gpu": 4,
  "gradient_accumulation_steps": 1,
  "steps_per_print": 100,
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 2e-4,
      "betas": [0.9, 0.999],
      "eps": 1e-8,
      "weight_decay": 0.01
    }
  },
  "scheduler": {
    "type": "WarmupDecayLR",
    "params": {
      "total_num_steps": 5000,
      "warmup_min_lr": 0,
      "warmup_max_lr": 2e-4,
      "warmup_num_steps": 500
    }
  },
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 50000000,
    "allgather_bucket_size": 50000000
  },
  "gradient_clipping": 1.0,
  "wall_clock_breakdown": true,
  "logging": {
    "logging_freq": 10
  }
}
EOF

cat > train/config/zero3_a100.yaml << 'EOF'
compute_environment: LOCAL_MACHINE
debug: false
deepspeed_config:
  deepspeed_config_file: /llamaSFT/askGuru-SQL/train/config/dp_zero3.json
  zero3_init_flag: true
distributed_type: DEEPSPEED
downcast_bf16: 'no'
enable_cpu_affinity: false
machine_rank: 0
main_training_function: train
num_machines: 1
num_processes: 8
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF
```

### 2. Create Training Arguments Config

```bash
cat > train_config.yaml << 'EOF'
# Model configuration
model_name_or_path: /llamaSFT/models/llama-3.3-70b-instruct
model_max_length: 8192

# Data configuration
data_path: /llamaSFT/data/oracle_sft_conversations/oracle_sft_conversations_train.json
eval_data_path: /llamaSFT/data/oracle_sft_conversations/oracle_sft_conversations_val.json

# Training configuration
output_dir: /llamaSFT/outputs/oracle_llama70b_lora
num_train_epochs: 3
per_device_train_batch_size: 4
per_device_eval_batch_size: 8
gradient_accumulation_steps: 1
learning_rate: 2.0e-4
warmup_steps: 500
logging_steps: 100
save_steps: 500
save_total_limit: 3
eval_steps: 500
eval_strategy: steps
save_strategy: steps
metric_for_best_model: loss
seed: 42

# LoRA configuration
use_lora: true
lora_r: 32
lora_alpha: 32
lora_dropout: 0.05
lora_target_modules: q_proj,k_proj,v_proj,o_proj,up_proj,gate_proj,down_proj

# Optimization
optim: adamw_8bit
lr_scheduler_type: linear
gradient_checkpointing: true
bf16: true
max_grad_norm: 1.0

# Inference settings
enable_dialect_router: false
temperature: 0.0

# Logging
logging_dir: /llamaSFT/logs/
logging_strategy: steps
report_to: [tensorboard]

# Resume
resume_from_checkpoint: null
EOF
```

---

## Running Fine-tuning

### 1. Start Training with Accelerate

```bash
# Ensure venv is activated
cd /llamaSFT
source venv/bin/activate

# Run training with DeepSpeed ZeRO-3
accelerate launch --config_file train/config/zero3_a100.yaml \
  askGuru-SQL/custom_oracle_llama/sft_oracle_llama70b_lora.py \
  --model_name_or_path /llamaSFT/models/llama-3.3-70b-instruct \
  --data_path /llamaSFT/data/oracle_sft_conversations/oracle_sft_conversations_train.json \
  --eval_data_path /llamaSFT/data/oracle_sft_conversations/oracle_sft_conversations_val.json \
  --output_dir /llamaSFT/outputs/oracle_llama70b_lora \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --learning_rate 2.0e-4 \
  --warmup_steps 500 \
  --logging_steps 100 \
  --save_steps 500 \
  --eval_steps 500 \
  --eval_strategy steps \
  --save_strategy steps \
  --model_max_length 8192 \
  --use_lora True \
  --q_lora False \
  --enable_dialect_router False \
  --lora_r 32 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,up_proj,gate_proj,down_proj \
  --gradient_checkpointing true \
  --bf16 true \
  --optim adamw_8bit \
  --max_grad_norm 1.0 \
  --seed 42 \
  --logging_dir /llamaSFT/logs/ \
  --report_to tensorboard \
  2>&1 | tee /llamaSFT/logs/training.log
```

### 2. Alternative: Run in Background (tmux)

```bash
# Start a tmux session
tmux new-session -d -s training -x 200 -y 50

# Send training command to tmux
tmux send-keys -t training "cd /llamaSFT && source venv/bin/activate" Enter
tmux send-keys -t training "accelerate launch --config_file train/config/zero3_a100.yaml custom_oracle_llama/sft_oracle_llama70b_lora.py ..." Enter

# Monitor training
tmux attach-session -t training

# Detach with Ctrl+B, then D
```

### 3. Monitor Training Progress

**Option 1: Watch GPU Usage in Real-time**

```bash
# In a separate terminal
watch -n 1 nvidia-smi
# Or use nvtop for better visualization
nvtop
```

**Option 2: Monitor Training Metrics**

```bash
# In a separate terminal
cd /llamaSFT
source venv/bin/activate
tensorboard --logdir /llamaSFT/logs/ --port 6006
# Access at http://localhost:6006
```

**Option 3: Monitor System Resources**

```bash
# In a separate terminal
htop  # System resources
iotop # Disk I/O (sudo required)
```

---

## Monitoring & Troubleshooting

### 1. Check Training Logs

```bash
# Real-time log monitoring
tail -f /llamaSFT/logs/training.log

# Search for errors
grep -i "error\|exception" /llamaSFT/logs/training.log

# Check loss progression
grep "loss" /llamaSFT/logs/training.log | tail -20
```

### 2. Common Issues & Solutions

**Issue: CUDA Out of Memory (OOM)**

```bash
# Reduce per_device_train_batch_size
--per_device_train_batch_size 2  # instead of 4

# Enable more aggressive gradient checkpointing
--gradient_checkpointing true

# Reduce model_max_length
--model_max_length 4096  # instead of 8192
```

**Issue: NVLink Not Detected**

```bash
# Check NVLink connectivity
nvidia-smi topo -m

# If NVLinks unavailable, training will be slower but should still work
```

**Issue: Training Hangs or Stalls**

```bash
# Check process status
ps aux | grep accelerate
ps aux | grep python

# Restart training from checkpoint
accelerate launch --config_file train/config/zero3_a100.yaml \
  custom_oracle_llama/sft_oracle_llama70b_lora.py \
  ... \
  --resume_from_checkpoint outputs/oracle_llama70b_lora/checkpoint-500
```

### 3. Estimate Training Time

```bash
# Rough estimate calculation:
# - 3,857 training examples
# - Batch size: 32 (4 per GPU × 8 GPUs)
# - 3 epochs ≈ 362 steps per epoch
# - ~1,086 total steps
# - ~0.2-0.3 seconds per step = ~3.5-5.5 hours total
# 
# With A100s at full utilization: ~4-6 hours for 3 epochs
```

---

## Output & Post-Training

### 1. Check Training Outputs

```bash
# List checkpoints and final model
ls -lh outputs/oracle_llama70b_lora/

# Expected structure:
# - checkpoint-500/
# - checkpoint-1000/
# - adapter_config.json
# - adapter_model.bin
# - training_args.bin
# - optimizer.pt (with ZeRO-3)
```

### 2. Validate LoRA Adapter

```bash
python3 << 'EOF'
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "models/llama-3.3-70b-instruct",
    torch_dtype="auto",
    device_map="auto"
)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "outputs/oracle_llama70b_lora")
print("✓ LoRA adapter loaded successfully")
print(f"  Adapter r: {model.peft_config['default'].r}")
print(f"  Adapter alpha: {model.peft_config['default'].lora_alpha}")
EOF
```

### 3. Merge LoRA into Base Model (Optional)

```bash
# Install dependencies for merging
pip install -r requirements.txt (not needed as we have done this already earlier)

#but install the following
pip install autoawq --no-build-isolation

#if that fails, then try this
pip install autoawq==0.2.8 --no-build-isolation --extra-index-url https://download.pytorch.org/whl/cu124

# Install directly from source
pip install git+https://github.com/AutoGPTQ/AutoGPTQ.git --no-build-isolation

pip install vllm>=0.6.0

sudo apt-get update && sudo apt-get install libaio-dev -y


# Run merge script
export PYTHONPATH=$PYTHONPATH:/llamaSFT/askGuru-SQL
python askGuru-SQL/custom_oracle_llama/package_oracle_model.py \
  --base_model models/llama-3.3-70b-instruct \
  --lora_adapter outputs/oracle_llama70b_lora \
  --merged_out outputs/merged_oracle_llama70b \
  --quant_out outputs/merged_oracle_llama70b_awq4 \
  --quant_method awq
```

### 4. Save Training Report

```bash
# Generate training summary
python3 << 'EOF'
import json
import os
from datetime import datetime

report = {
    "timestamp": datetime.now().isoformat(),
    "model": "LLaMA-3.3-70B-Instruct",
    "training_type": "LoRA (BF16)",
    "hardware": "8× A100-80GB",
    "training_data": {
        "train_examples": 3857,
        "val_examples": 482,
        "test_examples": 483
    },
    "training_config": {
        "learning_rate": 2.0e-4,
        "num_epochs": 3,
        "batch_size": 32,
        "lora_r": 32,
        "lora_alpha": 32
    },
    "output": {
        "lora_adapter": "outputs/oracle_llama70b_lora",
        "checkpoint_dir": "outputs/oracle_llama70b_lora",
        "log_dir": "logs/"
    }
}

with open("training_report.json", "w") as f:
    json.dump(report, f, indent=2)

print("✓ Training report saved to training_report.json")
EOF
```

---

## Next Steps

1. **Merge & Quantize Model** → See `DEPLOY_INFERENCE_UBUNTU24.md`
2. **Deploy with vLLM** → See `DEPLOY_INFERENCE_UBUNTU24.md`
3. **Evaluate on Test Set** → Run `evaluation/sql_eval.py`

---

## References

- **LLaMA Model:** https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct
- **DeepSpeed:** https://github.com/microsoft/DeepSpeed
- **PEFT/LoRA:** https://github.com/huggingface/peft
- **Accelerate:** https://huggingface.co/docs/accelerate
- **Flash Attention:** https://github.com/Dao-AILab/flash-attention

---

