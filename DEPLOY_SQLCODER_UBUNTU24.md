# SQLCoder-70B Fine-Tuning & Inference Deployment Guide

**Ubuntu 24 LTS on 8×A100-80GB (training) and 1×A100-80GB (inference)**

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Fine-tuning (8×A100)](#fine-tuning-8a100)
4. [Inference with vLLM (1×A100)](#inference-with-vllm-1a100)
5. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Hardware Requirements

**Fine-tuning:**
- 8× NVIDIA A100-80GB (preferred with NVLink)
- 128+ CPU cores
- 1TB+ RAM
- 500GB storage (model + data + checkpoints)

**Inference:**
- 1× NVIDIA A100-80GB
- 32+ CPU cores
- 256GB+ RAM
- 300GB storage (model + quantization)

### Software

- Ubuntu 24.04 LTS
- NVIDIA Driver 570+
- CUDA 12.8
- Python 3.12
- Git

---

## Environment Setup

### 1. Install System Dependencies

```bash
sudo apt-get update && sudo apt-get upgrade -y

sudo apt-get install -y \
  build-essential python3.12 python3.12-dev python3.12-venv \
  python3-pip git wget curl htop nvtop tmux openssh-server
```

### 2. Create Virtual Environment

```bash
cd /llamaSFT/askGuru-SQL
python3.12 -m venv .venv
source .venv/bin/activate
```

### 3. Install PyTorch with CUDA 12.8

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Verify
python -c "import torch; print(torch.cuda.is_available())"
```

### 4. Install askGuru-SQL Dependencies

```bash
cd /llamaSFT

# Setup HF_HOME
mkdir -p /llamaSFT/hf_home
export HF_HOME=/llamaSFT/hf_home
echo 'export HF_HOME=/llamaSFT/hf_home' >> /home/ubuntu/.bashrc

pip install -e .

# Core dependencies
pip install transformers datasets peft bitsandbytes accelerate deepspeed

# Training
pip install wandb trl

# Inference
pip install vllm

# Development
pip install black ruff mypy pytest
```

### 5. HuggingFace Setup

```bash
huggingface-cli login
# Paste your HF token (needs access to defog/sqlcoder-70b-alpha)
```

---

## Fine-tuning (8×A100)

### 1. Prepare Data

```bash
cd /llamaSFT

# Dataset already exists from LLaMA training
ls -lh /llamaSFT/data/oracle_sft_conversations/

# Expected output:
# -rw-r--r--  oracle_sft_conversations_train.json   (3,857 examples)
# -rw-r--r--  oracle_sft_conversations_val.json     (482 examples)
# -rw-r--r--  oracle_sft_conversations_test.json    (483 examples)
```

### 2. Download Model

```bash
# Download once, reuse for all GPUs
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = 'defog/sqlcoder-70b-alpha'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map='cpu'
)
print('✓ Model downloaded successfully')
"
```

### 3. Start Training

```bash
# Set visible GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Run training with Accelerate
cd /llamaSFT
accelerate launch --config_file train/config/zero3.yaml \
  custom_oracle_sqlcoder/sft_oracle_sqlcoder70b_lora.py \
  --model_name_or_path defog/sqlcoder-70b-alpha \
  --data_path /llamaSFT/data/oracle_sft_conversations/oracle_sft_conversations_train.json \
  --eval_data_path /llamaSFT/data/oracle_sft_conversations/oracle_sft_conversations_val.json \
  --output_dir /llamaSFT/outputs/oracle_sqlcoder70b_lora \
  --model_max_length 4096 \
  --use_lora True \
  --q_lora False \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2.0e-4 \
  --warmup_steps 500 \
  --num_train_epochs 3 \
  --save_steps 500 \
  --eval_steps 500 \
  --logging_steps 100 \
  --bf16 True \
  --seed 42 \
  --report_to wandb \
  --run_name "oracle-sqlcoder70b-lora"
```

**Expected output:**
```
Step 100: loss = 0.847
Step 200: loss = 0.623
Step 300: loss = 0.512
...
Training completed. Final loss: 0.284
✓ Model saved to outputs/oracle_sqlcoder70b_lora
```

**Training time:** ~8-10 hours for 3 epochs

### 4. Monitor Training

In another terminal:

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Watch memory/CPU
htop

# View W&B dashboard
# https://wandb.ai/your-username/oracle-sqlcoder70b-lora
```

### 5. Resume Training (if interrupted)

```bash
# Find latest checkpoint
ls /llamaSFT/outputs/oracle_sqlcoder70b_lora/

# Example: checkpoint-1500

cd /llamaSFT
accelerate launch --config_file train/config/zero3.yaml \
  custom_oracle_sqlcoder/sft_oracle_sqlcoder70b_lora.py \
  ... (same as above) \
  --resume_from_checkpoint /llamaSFT/outputs/oracle_sqlcoder70b_lora/checkpoint-1500
```

---

## Inference with vLLM (1×A100)

### 1. Merge LoRA Weights

```bash
# Create merged model (LoRA + base weights combined)

python << 'EOF'
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

base_model_id = "defog/sqlcoder-70b-alpha"
lora_model_path = "/llamaSFT/outputs/oracle_sqlcoder70b_lora"

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

model = PeftModel.from_pretrained(model, lora_model_path)
merged_model = model.merge_and_unload()

merged_model.save_pretrained("/llamaSFT/models/oracle_sqlcoder70b_merged")
tokenizer.save_pretrained("/llamaSFT/models/oracle_sqlcoder70b_merged")

print("✓ Model merged successfully")
EOF
```

### 2. (Optional) Quantize to 4-bit

```bash
python << 'EOF'
# Requires: pip install auto-gptq

from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

model_path = "/llamaSFT/models/oracle_sqlcoder70b_merged"

quantize_config = GPTQConfig(bits=4, desc_act=False, dataset="wikitext2")

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=quantize_config,
    device_map="auto"
)

model.save_pretrained("/llamaSFT/models/oracle_sqlcoder70b_4bit")

print("✓ Model quantized to 4-bit")
EOF
```

### 3. Create vLLM Config

```bash
cat > custom_oracle_sqlcoder/vllm_config.yaml << 'EOF'
model: /llamaSFT/models/oracle_sqlcoder70b_merged

# Inference parameters
max_model_len: 4096
max_num_batched_tokens: 8192
max_num_seqs: 4

# Generation
temperature: 0.1
top_p: 0.95
max_tokens: 512

# Optimization
gpu_memory_utilization: 0.85
tensor_parallel_size: 1
dtype: bfloat16
disable_log_stats: false
EOF
```

### 4. Start vLLM Server

```bash
# Foreground (for testing)
python -m vllm.entrypoints.openai.api_server \
  --model models/oracle_sqlcoder70b_merged \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --tensor-parallel-size 1 \
  --port 8000

# Or background with tmux
tmux new-session -d -s vllm \
  "python -m vllm.entrypoints.openai.api_server \
    --model models/oracle_sqlcoder70b_merged \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 4096 \
    --port 8000"

# Verify
sleep 5
curl http://localhost:8000/v1/models
```

### 5. Test Inference

**Using Python SDK:**
```python
from custom_oracle_sqlcoder.inference_oracle_sqlcoder import SQLCoderInference

inference = SQLCoderInference(
    model_path="models/oracle_sqlcoder70b_merged",
    max_tokens=512
)

schema = """
CREATE TABLE suppliers (supplier_id NUMBER, supplier_name VARCHAR2(240));
CREATE TABLE invoices (invoice_id NUMBER, supplier_id NUMBER, amount NUMBER);
"""

question = "List suppliers with total invoice amount > 100000"

sql = inference.generate(question, schema, validate_joins=True)
print(sql)
```

**Using OpenAI-compatible API:**
```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "models/oracle_sqlcoder70b_merged",
    "prompt": "### Task\nGenerate SQL to list suppliers\n\n### Database Schema\nCREATE TABLE suppliers...\n\n### SQL Query\n",
    "temperature": 0.1,
    "max_tokens": 512
  }'
```

---

## Troubleshooting

### Out of Memory (OOM)

**Error**: `CUDA out of memory`

**Solution:**
```bash
# Reduce batch size
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 16  # Effective batch: 2*16*8=256

# Or reduce model_max_length
--model_max_length 2048
```

### Model Download Fails

```bash
# Check HF credentials
huggingface-cli whoami

# Force re-download
rm -rf ~/.cache/huggingface/hub/defog*
huggingface-cli download defog/sqlcoder-70b-alpha
```

### vLLM Server Crashes

```bash
# Check logs
tail -f logs/vllm.log

# Restart with verbose logging
python -m vllm.entrypoints.openai.api_server \
  --model models/oracle_sqlcoder70b_merged \
  --verbose
```

### Poor SQL Quality

1. Check training completed successfully
2. Verify LoRA was merged correctly
3. Test with simple queries first
4. Enable `validate_joins=True` to catch errors

---

## Performance Expectations

| Metric | Value |
|--------|-------|
| **Training** | |
| Batch size (effective) | 256 (4*8*8 GPUs) |
| Tokens/second | 8-10K |
| Time per epoch | 2.5-3 hours |
| Total time (3 epochs) | 8-10 hours |
| LoRA adapter size | ~300MB |
| | |
| **Inference** | |
| Throughput | 50-100 tokens/sec |
| Latency (512 tokens) | 5-10 seconds |
| Concurrent requests | 4 |
| GPU memory | 55-60GB |

---

## Next Steps

1. **Comparison**: Compare SQLCoder output with LLaMA-3.1-70B
2. **Validation**: Test both models on your test set (483 examples)
3. **Decision**: Keep both or stick with LLaMA-only?
4. **Deployment**: If keeping both, implement routing logic

---

## References

- [Defog SQLCoder GitHub](https://github.com/defog-ai/sqlcoder)
- [vLLM Serving](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)
- [DeepSpeed Documentation](https://www.deepspeed.ai/)
