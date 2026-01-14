# custom_oracle_llama

Oracle EBS NL2SQL training and inference adapters for **askGuru-SQL** (formerly askGuru-SQL).

This folder contains **additive** scripts to fine-tune and deploy **LLaMA-3.3-70B** or **LLaMA-3.1-70B-Instruct** for Oracle EBS natural language to SQL conversion:

- **Training:** 8×A100-80GB (LoRA BF16 + Deepspeed ZeRO-3)
- **Inference:** 1×A100-80GB (4-bit quantized, 4 concurrent users)

> **Note**: **LLaMA-3.3-70B** is the recommended primary model due to its superior reasoning (comparable to 3.1-405B) and native 128K context window.

## Files

### Dataset & Training

- **`build_oracle_sft_dataset.py`**  
  Convert raw JSON/JSONL into askGuru `conversations` SFT format with RAG context.

- **`sft_oracle_llama70b_lora.py`**  
  Main training entrypoint (LoRA BF16, recommended for production).

- **`sft_oracle_llama70b_qlora.py`** (optional)  
  QLoRA training entrypoint (4-bit base, for limited GPU scenarios).

- **`train_util_4bit.py`** (optional)  
  Utilities to load model in 4-bit for QLoRA experiments.

### Packaging

- **`package_oracle_model.py`**  
  Merge LoRA adapters into base model, then quantize to 4-bit (AWQ/GPTQ).

### Inference

- **`inference/vllm_config.yaml`**  
  vLLM serving configuration for 1×A100-80GB deployment (4 concurrent).

- **`inference/sql_guardrail.py`**  
  SQL validation, cleanup, and one-retry prompt builder.

## Typical Workflow

### 1. Prepare Dataset

```bash
python custom_oracle_llama/build_oracle_sft_dataset.py \
  --input_raw data/oracle_raw.jsonl \
  --output_sft data/oracle_sft_conversations.json \
  --dataset_id oracle_ebs_v1
```

**Input format** (raw data):
```json
{
  "question": "How many invoices were paid in AP organization?",
  "sql_gold": "SELECT COUNT(*) FROM ap_invoices WHERE org_id = 1 AND status = 'PAID'",
  "rag_context": "# Tables\nap_invoices (org_id, invoice_id, status)\n# Joins\n..."
}
```

### 2. Train (LoRA BF16, recommended)

```bash
accelerate launch --config_file train/config/zero3.yaml \
  custom_oracle_llama/sft_oracle_llama70b_lora.py \
  --model_name_or_path /models/llama-3.3-70b-instruct \
  --data_path data/oracle_sft_conversations.json \
  --output_dir outputs/oracle_llama70b_lora \
  --model_max_length 8192 \
  --use_lora True \
  --q_lora False \
  --enable_dialect_router False
```

**Or, optional QLoRA** (if limited GPU):

```bash
accelerate launch --config_file train/config/zero3.yaml \
  custom_oracle_llama/sft_oracle_llama70b_qlora.py \
  --model_name_or_path /models/llama-3.3-70b-instruct \
  --data_path data/oracle_sft_conversations.json \
  --output_dir outputs/oracle_llama70b_qlora \
  --model_max_length 8192
```

### 3. Merge & Quantize

```bash
python custom_oracle_llama/package_oracle_model.py \
  --base_model /models/llama-3.3-70b-instruct \
  --lora_adapter outputs/oracle_llama70b_lora \
  --merged_out outputs/merged_oracle_llama70b \
  --quant_out outputs/merged_oracle_llama70b_awq4 \
  --quant_method awq
```

**Or with GPTQ:**

```bash
python custom_oracle_llama/package_oracle_model.py \
  --base_model /models/llama-3.3-70b-instruct \
  --lora_adapter outputs/oracle_llama70b_lora \
  --merged_out outputs/merged_oracle_llama70b \
  --quant_out outputs/merged_oracle_llama70b_gptq4 \
  --quant_method gptq
```

### 4. Serve with vLLM

```bash
python -m vllm.entrypoints.openai.api_server \
  --model outputs/merged_oracle_llama70b_awq4 \
  --max-model-len 8192 \
  --max-num-seqs 4 \
  --max-tokens 512 \
  --gpu-memory-utilization 0.92
```

Or load config from YAML:
```bash
python -m vllm.entrypoints.openai.api_server \
  --model outputs/merged_oracle_llama70b_awq4 \
  --config-file custom_oracle_llama/inference/vllm_config.yaml
```

### 5. Client + SQL Guardrail

```python
from custom_oracle_llama.inference.sql_guardrail import SQLGuardrail, validate_sql

guardrail = SQLGuardrail(allow_dml=False)

# After vLLM generation:
result = guardrail.validate(generated_text)
if result.ok:
    sql = result.cleaned_sql
else:
    retry_prompt = guardrail.retry_prompt(
        original_prompt, 
        result.cleaned_sql, 
        result.reason
    )
    # Send retry_prompt back to model for regeneration
```

## Configuration

### Training Defaults

- **Model max length:** 8192 tokens
- **Precision:** BF16 (base) or 4-bit quantized (QLoRA)
- **LoRA target modules:** `q_proj,k_proj,v_proj,o_proj,up_proj,gate_proj,down_proj`
- **LoRA rank (r):** 32 (adjustable; common range: 16–64)
- **LoRA alpha:** 32 (usually same as r)
- **LoRA dropout:** 0.05
- **Deepspeed:** ZeRO Stage 3 + Gradient Checkpointing
- **Dialect router:** disabled (Oracle-only)

### Inference Defaults

- **Quantization:** AWQ 4-bit (preferred) or GPTQ 4-bit
- **Max model length:** 8192
- **Max concurrent sequences:** 4
- **Max output tokens:** 512
- **Temperature:** 0.0 (deterministic SQL)
- **GPU memory utilization:** 0.92

## Design Notes

### No askGuru Overwrites
- ✅ Reuses: `train/trainer/trainer.py`, `train/trainer/train_util.py`, `train/utils/adapter_merge.py`
- ✅ Avoids: Multi-dialect routing, MOMQ paths (Oracle-only)
- ✅ Additive: All new code in `custom_oracle_llama/`

### SQL Validation
- **Guardian:** `sql_guardrail.py` enforces "SQL only" (no DML/DDL by default)
- **Retry:** One automatic retry with error feedback if validation fails
- **Hook:** Optional Oracle EXPLAIN/parse validation via pluggable hook

### Prompt Format
Structured user prompts with:
1. **Rules** — Oracle SQL dialect + no markdown/explanations
2. **RAG context** — Candidate tables, joins, columns, business mappings
3. **Question** — User's natural language request

## Requirements

Install dependencies:

```bash
pip install transformers datasets accelerate peft bitsandbytes
```

For quantization:
- **AWQ:** `pip install autoawq` (or `awq`)
- **GPTQ:** `pip install auto-gptq`

For serving:
- **vLLM:** `pip install vllm`

## Troubleshooting

### OOM during training
- Reduce `model_max_length` (e.g., 4096)
- Enable gradient checkpointing (default: on)
- Reduce batch size via `per_device_train_batch_size`

### OOM during inference
- Reduce `max_model_len` in vLLM config
- Reduce `max_num_seqs` (concurrent sequences)
- Reduce `gpu_memory_utilization` (e.g., 0.85)

### Quantization fails
- Ensure AWQ/GPTQ is installed: `pip install autoawq` or `pip install auto-gptq`
- Check merged model can load: `from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained(...)`

## References

- **askGuru-SQL:** Original framework [link]
- **askGuru-SQL:** Adapted project for Oracle EBS + Apache 2.0
- **LLaMA-3.1-70B:** Base model [HuggingFace](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct)
- **vLLM:** [https://github.com/vllm-project/vllm](https://github.com/vllm-project/vllm)
- **PEFT/LoRA:** [https://github.com/huggingface/peft](https://github.com/huggingface/peft)

## License

Apache 2.0 (inherited from askGuru-SQL)
