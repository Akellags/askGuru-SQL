#!/bin/bash

################################################################################
#
# deploy_finetune.sh
#
# Automated fine-tuning deployment script for Oracle EBS NL2SQL on 8×A100-80GB
# Ubuntu 24.04 LTS
#
# Usage:
#   ./deploy_finetune.sh [--skip-deps] [--skip-model] [--resume-from CHECKPOINT]
#
# Options:
#   --skip-deps       Skip dependency installation (assume already installed)
#   --skip-model      Skip model download (assume already downloaded)
#   --resume-from     Resume from checkpoint (e.g., checkpoint-500)
#   --help            Show this help message
#
################################################################################

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="/llamaSFT"
WORKSPACE="/llamaSFT"
VENV_PATH="/llamaSFT/venv"
MODELS_DIR="/llamaSFT/models"
DATA_DIR="/llamaSFT/data"
OUTPUTS_DIR="/llamaSFT/outputs"
LOGS_DIR="/llamaSFT/logs"
CHECKPOINTS_DIR="/llamaSFT/checkpoints"

# Set HuggingFace cache
export HF_HOME="/llamaSFT/hf_home"

# Training parameters
MODEL_NAME="llama-3.1-70b-instruct"
NUM_EPOCHS=3
LR=2.0e-4
WARMUP_STEPS=500
LOG_STEPS=100
SAVE_STEPS=500
EVAL_STEPS=500
MODEL_MAX_LEN=8192
BATCH_SIZE=4
LORA_R=32
SEED=42

# Flags
SKIP_DEPS=false
SKIP_MODEL=false
RESUME_FROM=""

################################################################################
# Utility Functions
################################################################################

log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*"
}

print_header() {
    echo -e "\n${BLUE}============================================================${NC}"
    echo -e "${BLUE}  $*${NC}"
    echo -e "${BLUE}============================================================${NC}\n"
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        return 1
    fi
    return 0
}

################################################################################
# Argument Parsing
################################################################################

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-deps)
                SKIP_DEPS=true
                log_info "Skipping dependency installation"
                shift
                ;;
            --skip-model)
                SKIP_MODEL=true
                log_info "Skipping model download"
                shift
                ;;
            --resume-from)
                RESUME_FROM="$2"
                log_info "Will resume from checkpoint: $RESUME_FROM"
                shift 2
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --skip-deps       Skip dependency installation"
    echo "  --skip-model      Skip model download"
    echo "  --resume-from     Resume from checkpoint (e.g., checkpoint-500)"
    echo "  --help            Show this help message"
    echo ""
    echo "Example:"
    echo "  ./deploy_finetune.sh"
    echo "  ./deploy_finetune.sh --skip-deps --skip-model"
    echo "  ./deploy_finetune.sh --resume-from checkpoint-500"
}

################################################################################
# Main Deployment Functions
################################################################################

setup_workspace() {
    print_header "Setting up workspace"
    
    mkdir -p "${WORKSPACE}" "${MODELS_DIR}" "${DATA_DIR}" "${OUTPUTS_DIR}" "${LOGS_DIR}" "${CHECKPOINTS_DIR}"
    
    # Copy data if exists in PROJECT_ROOT
    if [ -d "${PROJECT_ROOT}/data/oracle_sft_conversations" ]; then
        log_info "Copying training data..."
        cp -r "${PROJECT_ROOT}/data/oracle_sft_conversations" "${DATA_DIR}/"
    fi
    
    log_success "Workspace created at: ${WORKSPACE}"
}

check_gpu() {
    print_header "Checking GPU availability"
    
    if ! check_command nvidia-smi; then
        log_error "nvidia-smi not found. Please install NVIDIA drivers."
        exit 1
    fi
    
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    log_info "Found ${GPU_COUNT} NVIDIA GPU(s)"
    
    if [ "${GPU_COUNT}" -lt 8 ]; then
        log_warning "Expected 8 GPUs, found ${GPU_COUNT}. Training may be slow."
    fi
    
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
}

setup_python_env() {
    print_header "Setting up Python environment"
    
    # Check Python version
    if ! check_command python3.12; then
        log_warning "Python 3.12 not found, trying python3"
        if ! check_command python3; then
            log_error "Python 3 not found"
            exit 1
        fi
        PYTHON_CMD="python3"
    else
        PYTHON_CMD="python3.12"
    fi
    
    log_info "Using Python: ${PYTHON_CMD}"
    ${PYTHON_CMD} --version
    
    # Create virtual environment
    if [ ! -d "${VENV_PATH}" ]; then
        log_info "Creating Python virtual environment..."
        ${PYTHON_CMD} -m venv "${VENV_PATH}"
    else
        log_info "Virtual environment already exists"
    fi
    
    # Activate venv
    source "${VENV_PATH}/bin/activate"
    log_success "Virtual environment activated"
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    log_success "pip, setuptools, wheel upgraded"
}

install_dependencies() {
    print_header "Installing dependencies"
    
    if [ "${SKIP_DEPS}" = true ]; then
        log_info "Skipping dependency installation (--skip-deps)"
        return
    fi
    
    source "${VENV_PATH}/bin/activate"
    
    log_info "Installing PyTorch with CUDA 12.1 support..."
    pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
        --index-url https://download.pytorch.org/whl/cu121
    
    log_info "Installing core dependencies..."
    pip install \
        transformers==4.42.3 \
        datasets==2.18.0 \
        accelerate==0.31.0 \
        deepspeed==0.12.0 \
        peft==0.11.1 \
        flash-attn==2.5.9.post1 \
        bitsandbytes==0.43.1 \
        numpy==1.26.4 \
        pandas==2.2.3 \
        protobuf==5.27.2 \
        sentencepiece==0.2.0
    
    log_info "Installing optional dependencies (logging, quantization)..."
    pip install \
        swanlab==0.6.0 \
        wandb \
        tensorboard \
        autoawq \
        auto-gptq \
        huggingface-hub
    
    log_success "All dependencies installed"
    
    # Verify installation
    python3 << 'EOF'
import torch
import transformers
import datasets
import accelerate
import deepspeed
import peft
print("\n✓ Core dependencies verified")
print(f"  PyTorch: {torch.__version__}")
print(f"  Transformers: {transformers.__version__}")
print(f"  Datasets: {datasets.__version__}")
print(f"  Accelerate: {accelerate.__version__}")
print(f"  DeepSpeed: {deepspeed.__version__}")
print(f"  PEFT: {peft.__version__}")
EOF
}

download_model() {
    print_header "Downloading LLaMA-3.1-70B-Instruct model"
    
    if [ "${SKIP_MODEL}" = true ]; then
        log_info "Skipping model download (--skip-model)"
        return
    fi
    
    source "${VENV_PATH}/bin/activate"
    
    MODEL_PATH="${MODELS_DIR}/llama-3.1-70b-instruct"
    
    if [ -d "${MODEL_PATH}" ] && [ -f "${MODEL_PATH}/model.safetensors" ]; then
        log_success "Model already exists at ${MODEL_PATH}"
        return
    fi
    
    log_info "Model size: ~140GB. This may take 30-60 minutes..."
    log_info "Downloading from Hugging Face (requires authentication)"
    
    if [ -z "${HF_TOKEN:-}" ]; then
        log_warning "HF_TOKEN not set. Visit https://huggingface.co/settings/tokens"
        read -p "Enter your Hugging Face token: " HF_TOKEN
        export HF_TOKEN
    fi
    
    huggingface-cli login --token "${HF_TOKEN}"
    
    huggingface-cli download meta-llama/Llama-3.1-70B-Instruct \
        --local-dir "${MODEL_PATH}" \
        --token "${HF_TOKEN}"
    
    log_success "Model downloaded to ${MODEL_PATH}"
}

prepare_training_data() {
    print_header "Preparing training data"
    
    source "${VENV_PATH}/bin/activate"
    
    TRAIN_FILE="${DATA_DIR}/oracle_sft_conversations/oracle_sft_conversations_train.json"
    VAL_FILE="${DATA_DIR}/oracle_sft_conversations/oracle_sft_conversations_val.json"
    
    if [ ! -f "${TRAIN_FILE}" ]; then
        log_error "Training data not found at ${TRAIN_FILE}"
        exit 1
    fi
    
    if [ ! -f "${VAL_FILE}" ]; then
        log_error "Validation data not found at ${VAL_FILE}"
        exit 1
    fi
    
    log_info "Validating dataset format..."
    python3 << EOF
import json

for split, path in [('train', '${TRAIN_FILE}'), ('val', '${VAL_FILE}')]:
    with open(path, 'r') as f:
        data = json.load(f)
    
    print(f"✓ {split.upper()}: {len(data)} examples")
    
    # Validate first example
    ex = data[0]
    assert 'conversations' in ex, "Missing 'conversations'"
    assert len(ex['conversations']) == 2, "Expected 2 turns"
    assert '[User Question]' in ex['conversations'][0]['content'], "Missing [User Question]"
    assert 'SELECT' in ex['conversations'][1]['content'].upper(), "Expected SQL"
EOF
    
    log_success "Dataset validation passed"
}

setup_deepspeed_config() {
    print_header "Setting up DeepSpeed ZeRO-3 configuration"
    
    CONFIG_DIR="${PROJECT_ROOT}/train/config"
    mkdir -p "${CONFIG_DIR}"
    
    # Create DeepSpeed config
    cat > "${CONFIG_DIR}/dp_zero3.json" << 'EOF'
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
  "wall_clock_breakdown": true
}
EOF
    
    # Create accelerate config
    cat > "${CONFIG_DIR}/zero3_a100.yaml" << 'EOF'
compute_environment: LOCAL_MACHINE
debug: false
deepspeed_config:
  deepspeed_config_file: train/config/dp_zero3.json
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
    
    log_success "DeepSpeed configuration created"
}

run_training() {
    print_header "Starting fine-tuning training"
    
    source "${VENV_PATH}/bin/activate"
    cd "${PROJECT_ROOT}"
    
    # Build training command
    TRAIN_CMD="accelerate launch --config_file train/config/zero3_a100.yaml \
        custom_oracle_llama/sft_oracle_llama70b_lora.py \
        --model_name_or_path ${MODELS_DIR}/llama-3.1-70b-instruct \
        --data_path ${DATA_DIR}/oracle_sft_conversations/oracle_sft_conversations_train.json \
        --eval_data_path ${DATA_DIR}/oracle_sft_conversations/oracle_sft_conversations_val.json \
        --output_dir ${OUTPUTS_DIR}/oracle_llama70b_lora \
        --num_train_epochs ${NUM_EPOCHS} \
        --per_device_train_batch_size ${BATCH_SIZE} \
        --per_device_eval_batch_size 8 \
        --gradient_accumulation_steps 1 \
        --learning_rate ${LR} \
        --warmup_steps ${WARMUP_STEPS} \
        --logging_steps ${LOG_STEPS} \
        --save_steps ${SAVE_STEPS} \
        --eval_steps ${EVAL_STEPS} \
        --eval_strategy steps \
        --save_strategy steps \
        --model_max_length ${MODEL_MAX_LEN} \
        --use_lora True \
        --q_lora False \
        --enable_dialect_router False \
        --lora_r ${LORA_R} \
        --lora_alpha ${LORA_R} \
        --lora_dropout 0.05 \
        --lora_target_modules q_proj,k_proj,v_proj,o_proj,up_proj,gate_proj,down_proj \
        --gradient_checkpointing true \
        --bf16 true \
        --optim adamw_8bit \
        --max_grad_norm 1.0 \
        --seed ${SEED} \
        --logging_dir ${LOGS_DIR}/ \
        --report_to tensorboard \
        --overwrite_output_dir"
    
    # Add resume checkpoint if specified
    if [ -n "${RESUME_FROM}" ]; then
        TRAIN_CMD="${TRAIN_CMD} --resume_from_checkpoint ${OUTPUTS_DIR}/oracle_llama70b_lora/${RESUME_FROM}"
        log_info "Resuming from checkpoint: ${RESUME_FROM}"
    fi
    
    # Log training command
    log_info "Training command:"
    echo "${TRAIN_CMD}" | tr ' ' '\n' | sed 's/^/  /'
    
    # Run training
    log_info "Training started at $(date)"
    eval "${TRAIN_CMD}" 2>&1 | tee "${LOGS_DIR}/training_$(date +%Y%m%d_%H%M%S).log"
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        log_success "Training completed successfully"
    else
        log_error "Training failed. Check logs for details."
        exit 1
    fi
}

monitor_training() {
    print_header "Setting up training monitoring"
    
    source "${VENV_PATH}/bin/activate"
    
    log_info "TensorBoard available at:"
    log_info "  tensorboard --logdir ${LOGS_DIR}/ --port 6006"
    log_info ""
    log_info "GPU monitoring:"
    log_info "  watch -n 1 nvidia-smi"
    log_info "  or"
    log_info "  nvtop"
}

final_summary() {
    print_header "Deployment Summary"
    
    echo "Training Configuration:"
    echo "  Model: ${MODEL_NAME}"
    echo "  Hardware: 8× A100-80GB"
    echo "  Training Method: LoRA BF16 + DeepSpeed ZeRO-3"
    echo "  Learning Rate: ${LR}"
    echo "  Num Epochs: ${NUM_EPOCHS}"
    echo "  Batch Size: ${BATCH_SIZE} per GPU (${BATCH_SIZE} × 8)"
    echo "  LoRA Rank: ${LORA_R}"
    echo ""
    echo "Workspace:"
    echo "  Root: ${WORKSPACE}"
    echo "  Data: ${DATA_DIR}"
    echo "  Models: ${MODELS_DIR}"
    echo "  Outputs: ${OUTPUTS_DIR}"
    echo "  Logs: ${LOGS_DIR}"
    echo ""
    echo "Estimated Training Time: 4-6 hours (3 epochs)"
    echo ""
    log_success "Deployment complete!"
}

################################################################################
# Main Execution
################################################################################

main() {
    parse_args "$@"
    
    print_header "Oracle EBS NL2SQL Fine-tuning Deployment (8×A100-80GB)"
    
    check_gpu
    setup_workspace
    setup_python_env
    install_dependencies
    download_model
    prepare_training_data
    setup_deepspeed_config
    
    read -p "Ready to start training? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Training cancelled by user"
        exit 0
    fi
    
    run_training
    monitor_training
    final_summary
}

if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    main "$@"
fi
