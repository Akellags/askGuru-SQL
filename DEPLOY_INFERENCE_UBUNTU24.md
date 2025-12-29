# Oracle EBS NL2SQL Inference with vLLM on 1×A100-80GB (Ubuntu 24)

**Complete manual deployment guide for serving fine-tuned LLaMA-3.1-70B with vLLM**

---

## Table of Contents

1. [Prerequisites & Hardware](#prerequisites--hardware)
2. [Environment Setup](#environment-setup)
3. [NVIDIA Driver & CUDA Installation](#nvidia-driver--cuda-installation)
4. [PyTorch & Dependencies](#pytorch--dependencies)
5. [Model Preparation (Merge & Quantize)](#model-preparation-merge--quantize)
6. [vLLM Server Deployment](#vllm-server-deployment)
7. [Client Integration & Testing](#client-integration--testing)
8. [Production Monitoring](#production-monitoring)
9. [Troubleshooting](#troubleshooting)

---

## Prerequisites & Hardware

### Hardware Requirements
- **GPU:** 1× NVIDIA A100-80GB
- **CPU:** 32+ cores recommended
- **RAM:** 256GB+ system RAM
- **Storage:**
  - 150GB for quantized model (AWQ 4-bit)
  - 50GB for temporary working space

### Network Requirements
- Port 8000 (vLLM OpenAI API default)
- Port 6006 (optional: monitoring)
- 10 Gbps+ network for concurrent users

### Ubuntu 24 OS Requirements
- Ubuntu 24.04 LTS
- Root or sudo access
- Kernel 6.5+

---

## Environment Setup

### 1. Update System Packages

```bash
sudo apt-get update
sudo apt-get upgrade -y

# Install essential tools
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

### 2. Create Deployment User (Optional but Recommended)

```bash
# Create dedicated user for inference
sudo useradd -m -s /bin/bash inference
sudo usermod -aG sudo inference
sudo -u inference -i

# From here on, all commands run as 'inference' user
cd ~
```

### 3. Create Project Directory Structure

```bash
cd /llamaSFT

# Create subdirectories
mkdir -p {models,outputs,logs,data}

# Clone or copy project files
git clone <repo-url> . || echo "Copy project files manually"
```

### 4. Create Python Virtual Environment

```bash
cd /llamaSFT/askGuru-SQL
python3.12 -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

---

## NVIDIA Driver & CUDA Installation

### 1. Install NVIDIA Drivers

```bash
# Add NVIDIA repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID | sed -e 's/\.//g')
wget https://developer.download.nvidia.com/compute/cuda/repos/$distribution/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb

# Install CUDA toolkit
sudo apt-get update
sudo apt-get install -y cuda-toolkit libcudnn8 libcudnn8-dev

# Update PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> /home/ubuntu/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> /home/ubuntu/.bashrc
source /home/ubuntu/.bashrc

# Set HuggingFace cache
mkdir -p /llamaSFT/hf_home
export HF_HOME=/llamaSFT/hf_home
echo 'export HF_HOME=/llamaSFT/hf_home' >> /home/ubuntu/.bashrc

# Verify
nvidia-smi
nvcc --version
```

### 2. Verify CUDA & GPU

```bash
python3 << 'EOF'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")
print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
EOF
```

---

## PyTorch & Dependencies

### 1. Install PyTorch with CUDA 12.8

```bash
source /llamaSFT/askGuru-SQL/.venv/bin/activate

# PyTorch 2.5+ with CUDA 12.4 wheels (backward compatible with CUDA 12.8)
pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu124

# Verify
python3 << 'EOF'
import torch
assert torch.cuda.is_available()
print("✓ PyTorch + CUDA ready")
EOF
```

### 2. Install Core Inference Dependencies

```bash
pip install \
  transformers==4.42.3 \
  vllm==0.4.3 \
  peft==0.11.1 \
  accelerate==0.31.0 \
  bitsandbytes==0.43.1 \
  numpy==1.26.4 \
  pandas==2.2.3 \
  protobuf==5.27.2 \
  sentencepiece==0.2.0
```

### 3. Install Quantization Libraries

```bash
# AWQ (recommended for inference)
pip install autoawq

# Optional: GPTQ alternative
pip install auto-gptq
```

### 4. Install Monitoring & API Tools

```bash
pip install \
  tensorboard \
  wandb \
  openai \
  requests
```

---

## Model Preparation (Merge & Quantize)

### 1. Obtain Fine-tuned LoRA Checkpoint

Ensure you have the LoRA adapter from fine-tuning:
```bash
# From fine-tuning server
# Expected path: outputs/oracle_llama70b_lora/

# If on different machine, transfer via:
rsync -avz user@training-server:training_workspace/outputs/oracle_llama70b_lora ./models/
```

### 2. Merge LoRA Adapter into Base Model

```bash
source /llamaSFT/askGuru-SQL/.venv/bin/activate
cd /llamaSFT

# Step 1: Download base model (if not already present)
python3 << 'EOF'
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "meta-llama/Llama-3.1-70B-Instruct"
output_dir = "./models/llama-3.1-70b-instruct"

# Download model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Save locally
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"✓ Model saved to {output_dir}")
EOF
```

### 3. Run Merge Script

```bash
# Use the provided merge script
python custom_oracle_llama/package_oracle_model.py \
  --base_model ./models/llama-3.1-70b-instruct \
  --lora_adapter ./models/oracle_llama70b_lora \
  --merged_out ./models/merged_oracle_llama70b \
  --quant_out ./models/merged_oracle_llama70b_awq4 \
  --quant_method awq \
  2>&1 | tee logs/merge_quant_$(date +%Y%m%d_%H%M%S).log
```

**What this does:**
1. Loads base LLaMA-3.1-70B-Instruct
2. Merges LoRA adapter weights
3. Quantizes to 4-bit using AWQ
4. Saves merged quantized model

**Expected Output:**
```
✓ Model merging complete
✓ AWQ quantization complete
✓ Quantized model saved to models/merged_oracle_llama70b_awq4
```

### 4. Verify Merged Model

```bash
python3 << 'EOF'
from transformers import AutoModelForCausalLM, AutoTokenizer

# Test loading quantized model
model = AutoModelForCausalLM.from_pretrained(
    "./models/merged_oracle_llama70b_awq4",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("./models/merged_oracle_llama70b_awq4")

print("✓ Quantized model loaded successfully")
print(f"  Model type: {type(model)}")
print(f"  Vocab size: {len(tokenizer)}")

# Test a simple generation
prompt = "SELECT * FROM"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=10)
generated = tokenizer.decode(outputs[0])
print(f"\n✓ Generation test passed")
print(f"  Prompt: {prompt}")
print(f"  Output: {generated}")
EOF
```

---

## vLLM Server Deployment

### 1. Prepare vLLM Configuration

```bash
# Create inference config
cat > inference_config.yaml << 'EOF'
# vLLM Inference Configuration for Oracle EBS NL2SQL
# 1× A100-80GB, 4 concurrent users

model_name_or_path: ./models/merged_oracle_llama70b_awq4

# Model parameters
dtype: bfloat16
max_model_len: 8192
trust_remote_code: true

# GPU optimization
gpu_memory_utilization: 0.92
enforce_eager: false

# Concurrency & performance
max_num_seqs: 4           # Max concurrent sequences
max_tokens: 512           # Max output tokens per request

# vLLM server parameters
host: 0.0.0.0
port: 8000
tensor_parallel_size: 1
pipeline_parallel_size: 1

# Quantization
quantization: awq

# Generation defaults
temperature: 0.0          # Deterministic SQL
top_p: 1.0
top_k: -1
max_tokens: 512

# Logging
log_requests: true
log_stats: true
EOF

log_info "vLLM configuration created"
```

### 2. Start vLLM Server (Foreground)

For development/testing:

```bash
source /llamaSFT/askGuru-SQL/.venv/bin/activate
cd /llamaSFT

python -m vllm.entrypoints.openai.api_server \
  --model ./models/merged_oracle_llama70b_awq4 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --max-num-seqs 4 \
  --max-tokens 512 \
  --gpu-memory-utilization 0.92 \
  --quantization awq \
  --host 0.0.0.0 \
  --port 8000 \
  --enforce-eager
```

Expected output:
```
INFO 01-15 10:30:45 llm_engine.py:160] Initializing an LLM engine with config:
...
INFO 01-15 10:30:52 api_server.py:498] Started server process with PID 12345
Uvicorn running on http://0.0.0.0:8000
```

### 3. Start vLLM Server (Background/Production)

Using tmux:

```bash
# Create tmux session
tmux new-session -d -s vllm -x 200 -y 50

# Start server
tmux send-keys -t vllm "cd /llamaSFT && source /llamaSFT/askGuru-SQL/.venv/bin/activate" Enter
tmux send-keys -t vllm "python -m vllm.entrypoints.openai.api_server ..." Enter

# Monitor
tmux attach -t vllm

# Detach: Ctrl+B, D
```

Using systemd:

```bash
# Create systemd service
sudo tee /etc/systemd/system/vllm-oracle.service > /dev/null << 'EOF'
[Unit]
Description=vLLM Oracle EBS NL2SQL Inference Server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=inference
WorkingDirectory=/home/inference/askguru-sql-inference
ExecStart=/home/inference/askguru-sql-inference/venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model ./models/merged_oracle_llama70b_awq4 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --max-num-seqs 4 \
  --max-tokens 512 \
  --gpu-memory-utilization 0.92 \
  --quantization awq \
  --port 8000

Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable vllm-oracle
sudo systemctl start vllm-oracle

# Check status
sudo systemctl status vllm-oracle

# View logs
sudo journalctl -u vllm-oracle -f
```

### 4. Health Check

```bash
# Check if server is running
curl http://localhost:8000/health

# Expected: 200 OK
```

---

## Client Integration & Testing

### 1. Basic Test with curl

```bash
# Test inference endpoint
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "merged_oracle_llama70b_awq4",
    "prompt": "SELECT * FROM",
    "max_tokens": 50,
    "temperature": 0.0
  }'
```

### 2. Python Client (OpenAI-compatible)

```bash
python3 << 'EOF'
from openai import OpenAI

# Connect to vLLM server
client = OpenAI(
    api_key="not-needed",
    base_url="http://localhost:8000/v1"
)

# Test inference
response = client.completions.create(
    model="merged_oracle_llama70b_awq4",
    prompt="SELECT COUNT(*) FROM ap_invoices WHERE status = 'PAID'",
    max_tokens=100,
    temperature=0.0
)

print("✓ vLLM inference successful")
print(f"Generated SQL: {response.choices[0].text}")
EOF
```

### 3. Full NL2SQL Test

```bash
python3 << 'EOF'
from openai import OpenAI
import json

client = OpenAI(
    api_key="not-needed",
    base_url="http://localhost:8000/v1"
)

# Complete prompt format (as used in training)
user_prompt = """You are a Text-to-SQL generator for Oracle EBS.
Return ONLY Oracle SQL. No markdown. No explanations. No comments.

# Candidate Tables
- AP_INVOICES: INVOICE_ID, VENDOR_ID, INVOICE_NUM, STATUS, AMOUNT
- AP_SUPPLIERS: VENDOR_ID, SEGMENT1, VENDOR_NAME

# Join Graph
- AP_INVOICES.VENDOR_ID = AP_SUPPLIERS.VENDOR_ID

# Relevant Columns
- AP_INVOICES.STATUS: VARCHAR2
- AP_INVOICES.AMOUNT: NUMBER
- AP_SUPPLIERS.VENDOR_NAME: VARCHAR2

[User Question]
Count total amount of paid invoices by supplier"""

response = client.completions.create(
    model="merged_oracle_llama70b_awq4",
    prompt=user_prompt,
    max_tokens=256,
    temperature=0.0,
    top_p=1.0
)

generated_sql = response.choices[0].text.strip()
print("Generated SQL:")
print(generated_sql)
EOF
```

### 4. Batch Inference for Test Set

```bash
python3 << 'EOF'
import json
from openai import OpenAI

client = OpenAI(api_key="not-needed", base_url="http://localhost:8000/v1")

# Load test dataset
with open("data/oracle_sft_conversations/oracle_sft_conversations_test.json") as f:
    test_data = json.load(f)

results = []

for i, example in enumerate(test_data[:10]):  # Test on first 10
    user_prompt = example['conversations'][0]['content']
    expected_sql = example['conversations'][1]['content']
    
    response = client.completions.create(
        model="merged_oracle_llama70b_awq4",
        prompt=user_prompt,
        max_tokens=256,
        temperature=0.0
    )
    
    generated_sql = response.choices[0].text.strip()
    
    results.append({
        "id": example.get('id'),
        "expected": expected_sql,
        "generated": generated_sql,
        "match": generated_sql.lower() == expected_sql.lower()
    })
    
    print(f"[{i+1}/10] Example {example.get('id')}: {'✓' if results[-1]['match'] else '✗'}")

# Summary
matches = sum(1 for r in results if r['match'])
print(f"\n✓ Results: {matches}/{len(results)} exact matches")

# Save results
with open("inference_test_results.json", "w") as f:
    json.dump(results, f, indent=2)
EOF
```

---

## Production Monitoring

### 1. GPU Monitoring

```bash
# Real-time GPU stats
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,temperature.gpu,power.draw --format=csv,noheader -l 1

# Or use nvtop
nvtop
```

### 2. vLLM Server Metrics

```bash
# Get server stats (if logging enabled)
curl http://localhost:8000/stats

# Monitor logs in real-time
journalctl -u vllm-oracle -f

# Or tmux session logs
tmux capture-pane -t vllm -p -S -200 | tail -100
```

### 3. Request Monitoring

```bash
# Monitor HTTP requests (requires tcpdump or similar)
tcpdump -i any -n 'port 8000' -A

# Simple request counter
while true; do
    echo "$(date): $(curl -s http://localhost:8000/health | grep -q ok && echo 'HEALTHY' || echo 'DOWN')"
    sleep 5
done
```

### 4. Performance Benchmarking

```bash
python3 << 'EOF'
import time
import json
from openai import OpenAI

client = OpenAI(api_key="not-needed", base_url="http://localhost:8000/v1")

# Load test examples
with open("data/oracle_sft_conversations/oracle_sft_conversations_test.json") as f:
    test_data = json.load(f)

# Benchmark
latencies = []
throughputs = []

for example in test_data[:20]:
    prompt = example['conversations'][0]['content']
    
    start = time.time()
    response = client.completions.create(
        model="merged_oracle_llama70b_awq4",
        prompt=prompt,
        max_tokens=256,
        temperature=0.0
    )
    latency = time.time() - start
    latencies.append(latency)
    
    tokens = len(response.choices[0].text.split())
    throughput = tokens / latency
    throughputs.append(throughput)

# Results
print(f"Average Latency: {sum(latencies)/len(latencies):.2f}s")
print(f"Min/Max Latency: {min(latencies):.2f}s / {max(latencies):.2f}s")
print(f"Average Throughput: {sum(throughputs)/len(throughputs):.0f} tokens/sec")
EOF
```

---

## Troubleshooting

### Issue: CUDA Out of Memory

```bash
# Reduce max_num_seqs
python -m vllm.entrypoints.openai.api_server \
  --model ./models/merged_oracle_llama70b_awq4 \
  --max-num-seqs 2 \
  --gpu-memory-utilization 0.85

# Or reduce max_model_len
--max-model-len 4096
```

### Issue: Slow Inference (< 20 tokens/sec)

```bash
# Check GPU utilization
nvidia-smi

# If GPU usage low:
# - Increase max_num_seqs (up to 4-8 for A100-80GB)
# - Check CPU bottleneck (ps aux | grep python)
# - Verify quantization loaded (should show ~20GB VRAM)

# If GPU maxed out:
# - Your model is using full capacity (expected)
# - Consider batching requests in client
```

### Issue: Server Crashes on Startup

```bash
# Check vLLM version compatibility
pip show vllm

# Reinstall vLLM
pip uninstall vllm
pip install vllm==0.4.3

# Check CUDA/cuDNN compatibility
python3 -c "import torch; print(torch.version.cuda)"

# Run with verbose output
python -m vllm.entrypoints.openai.api_server \
  --model ./models/merged_oracle_llama70b_awq4 \
  --verbose
```

### Issue: Connection Refused

```bash
# Check if server is running
ps aux | grep vllm

# Check port availability
netstat -tuln | grep 8000

# Try different port
python -m vllm.entrypoints.openai.api_server \
  --port 8001
```

### Issue: Poor SQL Quality

```bash
# Verify fine-tuned model is loaded (not base model)
curl http://localhost:8000/v1/models

# Check prompt format matches training data
# Expected marker: [User Question] (English, not 【用户问题】)

# Test with more tokens
--max-tokens 512

# Verify SQL guardrail (optional)
from custom_oracle_llama.inference.sql_guardrail import SQLGuardrail
guardrail = SQLGuardrail()
result = guardrail.validate(generated_sql)
```

---

## Next Steps

1. **API Integration** → Implement client in your application
2. **Load Testing** → Test with concurrent clients
3. **Monitoring** → Set up production metrics collection
4. **Scaling** → Multiple vLLM instances behind load balancer

---

## References

- **vLLM:** https://github.com/vllm-project/vllm
- **OpenAI API Compatibility:** https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
- **AWQ Quantization:** https://github.com/mit-han-lab/awq
- **LLaMA Model:** https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct

---

