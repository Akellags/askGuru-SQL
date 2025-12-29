#!/bin/bash

################################################################################
#
# deploy_inference.sh
#
# Automated inference deployment script for Oracle EBS NL2SQL on 1×A100-80GB
# Ubuntu 24.04 LTS
# 
# Serves fine-tuned LLaMA-3.1-70B model using vLLM with OpenAI-compatible API
#
# Usage:
#   ./deploy_inference.sh [--skip-deps] [--skip-merge] [--port PORT] [--test]
#
# Options:
#   --skip-deps       Skip dependency installation
#   --skip-merge      Skip model merge/quantization (assumes already done)
#   --port PORT       Set vLLM server port (default: 8000)
#   --test            Run inference tests after startup
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
OUTPUTS_DIR="/llamaSFT/outputs"
LOGS_DIR="/llamaSFT/logs"

# Set HuggingFace cache
export HF_HOME="/llamaSFT/hf_home"

# vLLM parameters
VLLM_PORT=8000
VLLM_HOST="0.0.0.0"
VLLM_DTYPE="bfloat16"
VLLM_MAX_MODEL_LEN=8192
VLLM_MAX_NUM_SEQS=4
VLLM_MAX_TOKENS=512
VLLM_GPU_MEMORY_UTIL=0.92
VLLM_TEMPERATURE=0.0

# Flags
SKIP_DEPS=false
SKIP_MERGE=false
RUN_TESTS=false

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

wait_for_port() {
    local port=$1
    local timeout=$2
    local elapsed=0
    
    while [ $elapsed -lt $timeout ]; do
        if nc -z localhost $port 2>/dev/null; then
            return 0
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done
    
    return 1
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
            --skip-merge)
                SKIP_MERGE=true
                log_info "Skipping model merge/quantization"
                shift
                ;;
            --port)
                VLLM_PORT="$2"
                log_info "Using port: $VLLM_PORT"
                shift 2
                ;;
            --test)
                RUN_TESTS=true
                log_info "Will run inference tests"
                shift
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
    echo "  --skip-merge      Skip model merge/quantization"
    echo "  --port PORT       Set vLLM server port (default: 8000)"
    echo "  --test            Run inference tests after startup"
    echo "  --help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./deploy_inference.sh"
    echo "  ./deploy_inference.sh --skip-deps --skip-merge"
    echo "  ./deploy_inference.sh --port 8001 --test"
}

################################################################################
# Main Deployment Functions
################################################################################

setup_workspace() {
    print_header "Setting up inference workspace"
    
    mkdir -p "${WORKSPACE}" "${MODELS_DIR}" "${OUTPUTS_DIR}" "${LOGS_DIR}"
    
    # Copy models if available
    if [ -d "${PROJECT_ROOT}/models" ]; then
        log_info "Copying models..."
        cp -r "${PROJECT_ROOT}/models"/* "${MODELS_DIR}/" 2>/dev/null || true
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
    
    if [ "${GPU_COUNT}" -lt 1 ]; then
        log_error "At least 1 GPU required for inference"
        exit 1
    fi
    
    if [ "${GPU_COUNT}" -eq 1 ]; then
        log_success "Single A100-80GB detected (optimal for inference)"
    fi
    
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
}

setup_python_env() {
    print_header "Setting up Python environment"
    
    # Check Python version
    if ! check_command python3.12; then
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
    
    source "${VENV_PATH}/bin/activate"
    log_success "Virtual environment activated"
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
}

install_dependencies() {
    print_header "Installing inference dependencies"
    
    if [ "${SKIP_DEPS}" = true ]; then
        log_info "Skipping dependency installation (--skip-deps)"
        return
    fi
    
    source "${VENV_PATH}/bin/activate"
    
    log_info "Installing PyTorch with CUDA 12.1 support..."
    pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
        --index-url https://download.pytorch.org/whl/cu121
    
    log_info "Installing vLLM and core dependencies..."
    pip install \
        vllm==0.4.3 \
        transformers==4.42.3 \
        accelerate==0.31.0 \
        peft==0.11.1 \
        bitsandbytes==0.43.1 \
        numpy==1.26.4 \
        pandas==2.2.3 \
        protobuf==5.27.2 \
        sentencepiece==0.2.0
    
    log_info "Installing quantization & API tools..."
    pip install \
        autoawq \
        auto-gptq \
        openai \
        requests
    
    log_success "All dependencies installed"
    
    # Verify vLLM
    python3 << 'EOF'
import vllm
print(f"✓ vLLM version: {vllm.__version__}")
EOF
}

verify_models() {
    print_header "Verifying model files"
    
    source "${VENV_PATH}/bin/activate"
    
    # Check for base model
    BASE_MODEL="${MODELS_DIR}/llama-3.1-70b-instruct"
    if [ ! -d "${BASE_MODEL}" ]; then
        log_warning "Base model not found at ${BASE_MODEL}"
        log_info "You will need to download it manually or from fine-tuning server"
    else
        log_success "Base model found"
    fi
    
    # Check for merged quantized model
    MERGED_MODEL="${MODELS_DIR}/merged_oracle_llama70b_awq4"
    if [ -d "${MERGED_MODEL}" ]; then
        log_success "Merged quantized model found at ${MERGED_MODEL}"
        return
    fi
    
    # Check for LoRA adapter
    LORA_ADAPTER="${MODELS_DIR}/oracle_llama70b_lora"
    if [ ! -d "${LORA_ADAPTER}" ]; then
        log_error "LoRA adapter not found at ${LORA_ADAPTER}"
        log_error "Please provide: ./models/oracle_llama70b_lora/"
        exit 1
    fi
    
    log_success "LoRA adapter found"
}

merge_and_quantize() {
    print_header "Merging LoRA adapter and quantizing model"
    
    if [ "${SKIP_MERGE}" = true ]; then
        log_info "Skipping model merge/quantization (--skip-merge)"
        return
    fi
    
    source "${VENV_PATH}/bin/activate"
    cd "${PROJECT_ROOT}"
    
    MERGED_MODEL="${MODELS_DIR}/merged_oracle_llama70b_awq4"
    
    if [ -d "${MERGED_MODEL}" ]; then
        log_success "Merged model already exists at ${MERGED_MODEL}"
        return
    fi
    
    log_info "This will take 15-30 minutes..."
    log_info "Running merge script..."
    
    python custom_oracle_llama/package_oracle_model.py \
        --base_model "${MODELS_DIR}/llama-3.1-70b-instruct" \
        --lora_adapter "${MODELS_DIR}/oracle_llama70b_lora" \
        --merged_out "${MODELS_DIR}/merged_oracle_llama70b" \
        --quant_out "${MERGED_MODEL}" \
        --quant_method awq \
        2>&1 | tee "${LOGS_DIR}/merge_quant_$(date +%Y%m%d_%H%M%S).log"
    
    if [ ! -d "${MERGED_MODEL}" ]; then
        log_error "Model merge/quantization failed"
        exit 1
    fi
    
    log_success "Model merge and quantization complete"
}

create_systemd_service() {
    print_header "Creating systemd service (optional)"
    
    SERVICE_FILE="/etc/systemd/system/vllm-oracle.service"
    
    # Check if already exists
    if [ -f "${SERVICE_FILE}" ]; then
        log_info "Service already exists at ${SERVICE_FILE}"
        return
    fi
    
    log_info "Creating systemd service file..."
    
    # Get current username
    CURRENT_USER=$(whoami)
    
    cat > /tmp/vllm-oracle.service << EOF
[Unit]
Description=vLLM Oracle EBS NL2SQL Inference Server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${CURRENT_USER}
WorkingDirectory=${WORKSPACE}
ExecStart=${VENV_PATH}/bin/python -m vllm.entrypoints.openai.api_server \\
  --model ${MODELS_DIR}/merged_oracle_llama70b_awq4 \\
  --dtype ${VLLM_DTYPE} \\
  --max-model-len ${VLLM_MAX_MODEL_LEN} \\
  --max-num-seqs ${VLLM_MAX_NUM_SEQS} \\
  --max-tokens ${VLLM_MAX_TOKENS} \\
  --gpu-memory-utilization ${VLLM_GPU_MEMORY_UTIL} \\
  --quantization awq \\
  --host ${VLLM_HOST} \\
  --port ${VLLM_PORT}

Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
    
    log_info "To install systemd service:"
    echo ""
    echo "  sudo cp /tmp/vllm-oracle.service ${SERVICE_FILE}"
    echo "  sudo systemctl daemon-reload"
    echo "  sudo systemctl enable vllm-oracle"
    echo "  sudo systemctl start vllm-oracle"
    echo ""
}

start_vllm_server() {
    print_header "Starting vLLM inference server"
    
    source "${VENV_PATH}/bin/activate"
    cd "${WORKSPACE}"
    
    MERGED_MODEL="${MODELS_DIR}/merged_oracle_llama70b_awq4"
    
    if [ ! -d "${MERGED_MODEL}" ]; then
        log_error "Merged model not found at ${MERGED_MODEL}"
        exit 1
    fi
    
    log_info "Starting vLLM server on port ${VLLM_PORT}..."
    log_info "Model: ${MERGED_MODEL}"
    log_info "Max concurrent sequences: ${VLLM_MAX_NUM_SEQS}"
    echo ""
    
    # Start vLLM in background
    python -m vllm.entrypoints.openai.api_server \
        --model "${MERGED_MODEL}" \
        --dtype "${VLLM_DTYPE}" \
        --max-model-len "${VLLM_MAX_MODEL_LEN}" \
        --max-num-seqs "${VLLM_MAX_NUM_SEQS}" \
        --max-tokens "${VLLM_MAX_TOKENS}" \
        --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTIL}" \
        --quantization awq \
        --host "${VLLM_HOST}" \
        --port "${VLLM_PORT}" \
        --enforce-eager \
        2>&1 | tee "${LOGS_DIR}/vllm_$(date +%Y%m%d_%H%M%S).log" &
    
    VLLM_PID=$!
    log_info "vLLM server started with PID: ${VLLM_PID}"
    
    # Wait for server to be ready
    log_info "Waiting for server to be ready (max 120 seconds)..."
    if wait_for_port "${VLLM_PORT}" 120; then
        log_success "vLLM server is running on http://localhost:${VLLM_PORT}"
    else
        log_error "Server failed to start. Check logs for details."
        kill $VLLM_PID 2>/dev/null || true
        exit 1
    fi
}

test_inference() {
    print_header "Testing inference"
    
    if [ "${RUN_TESTS}" = false ]; then
        log_info "Skipping inference tests (use --test to run)"
        return
    fi
    
    source "${VENV_PATH}/bin/activate"
    
    log_info "Running basic inference test..."
    
    python3 << 'EOF'
import json
import time
from openai import OpenAI

try:
    client = OpenAI(api_key="not-needed", base_url="http://localhost:8000/v1")
    
    # Test 1: Simple completion
    print("\n[Test 1] Basic SQL generation:")
    response = client.completions.create(
        model="merged_oracle_llama70b_awq4",
        prompt="SELECT * FROM",
        max_tokens=20,
        temperature=0.0
    )
    print(f"✓ Generated: {response.choices[0].text.strip()}")
    
    # Test 2: Full NL2SQL prompt
    print("\n[Test 2] Full NL2SQL example:")
    prompt = """You are a Text-to-SQL generator for Oracle EBS.
Return ONLY Oracle SQL. No markdown. No explanations. No comments.

# Candidate Tables
- AP_INVOICES: INVOICE_ID, VENDOR_ID, STATUS

[User Question]
Count paid invoices"""
    
    start = time.time()
    response = client.completions.create(
        model="merged_oracle_llama70b_awq4",
        prompt=prompt,
        max_tokens=100,
        temperature=0.0
    )
    latency = time.time() - start
    
    sql = response.choices[0].text.strip()
    print(f"✓ Generated SQL: {sql}")
    print(f"  Latency: {latency:.2f}s")
    print(f"  Tokens/sec: {len(sql.split())/latency:.0f}")
    
    # Test 3: Load test data
    print("\n[Test 3] Testing on sample dataset:")
    with open("../data/oracle_sft_conversations/oracle_sft_conversations_test.json") as f:
        test_data = json.load(f)[:5]
    
    correct = 0
    for i, example in enumerate(test_data):
        user_prompt = example['conversations'][0]['content']
        expected = example['conversations'][1]['content']
        
        response = client.completions.create(
            model="merged_oracle_llama70b_awq4",
            prompt=user_prompt,
            max_tokens=256,
            temperature=0.0
        )
        
        generated = response.choices[0].text.strip()
        is_match = generated.lower() == expected.lower()
        correct += is_match
        
        print(f"  [{i+1}/5] {example.get('id')}: {'✓' if is_match else '✗'}")
    
    print(f"\n✓ Inference tests passed ({correct}/5 exact matches)")
    
except Exception as e:
    print(f"✗ Test failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
EOF
}

show_usage_info() {
    print_header "vLLM Inference Server Ready"
    
    echo "Server Details:"
    echo "  URL: http://localhost:${VLLM_PORT}"
    echo "  Model: merged_oracle_llama70b_awq4"
    echo "  Max concurrent: ${VLLM_MAX_NUM_SEQS}"
    echo "  Max output: ${VLLM_MAX_TOKENS} tokens"
    echo ""
    echo "Quick Test Commands:"
    echo ""
    echo "1. Health check:"
    echo "   curl http://localhost:${VLLM_PORT}/health"
    echo ""
    echo "2. List models:"
    echo "   curl http://localhost:${VLLM_PORT}/v1/models"
    echo ""
    echo "3. Test inference:"
    echo "   curl http://localhost:${VLLM_PORT}/v1/completions \\"
    echo "     -H 'Content-Type: application/json' \\"
    echo "     -d '{\"model\":\"merged_oracle_llama70b_awq4\",\"prompt\":\"SELECT * FROM\",\"max_tokens\":50}'"
    echo ""
    echo "4. Python client:"
    echo "   from openai import OpenAI"
    echo "   client = OpenAI(api_key='not-needed', base_url='http://localhost:${VLLM_PORT}/v1')"
    echo "   response = client.completions.create(model='merged_oracle_llama70b_awq4', prompt='SELECT')"
    echo ""
    echo "Monitoring:"
    echo "  GPU: nvidia-smi -l 1"
    echo "  Logs: tail -f ${LOGS_DIR}/vllm_*.log"
    echo ""
    log_success "Server is ready to accept inference requests!"
}

final_summary() {
    print_header "Deployment Summary"
    
    echo "Configuration:"
    echo "  Model: merged_oracle_llama70b_awq4 (4-bit AWQ quantized)"
    echo "  Hardware: 1× A100-80GB"
    echo "  Port: ${VLLM_PORT}"
    echo "  Dtype: ${VLLM_DTYPE}"
    echo "  Max model length: ${VLLM_MAX_MODEL_LEN}"
    echo "  Max concurrent: ${VLLM_MAX_NUM_SEQS}"
    echo ""
    echo "Workspace:"
    echo "  Root: ${WORKSPACE}"
    echo "  Models: ${MODELS_DIR}"
    echo "  Logs: ${LOGS_DIR}"
    echo ""
    log_success "Inference deployment complete!"
}

################################################################################
# Main Execution
################################################################################

main() {
    parse_args "$@"
    
    print_header "Oracle EBS NL2SQL Inference Deployment (1×A100-80GB)"
    
    check_gpu
    setup_workspace
    setup_python_env
    install_dependencies
    verify_models
    merge_and_quantize
    create_systemd_service
    start_vllm_server
    test_inference
    show_usage_info
    final_summary
}

if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    main "$@"
fi
