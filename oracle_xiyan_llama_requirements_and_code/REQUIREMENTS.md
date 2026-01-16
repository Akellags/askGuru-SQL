# Oracle EBS NL2SQL Fine-tuning & Deployment (XiYan-SQLTraining + LLaMA-3.3-70B-Instruct)

## 1) Objective

Fine-tune **LLaMA-3.3-70B-Instruct** for **Oracle EBS NL2SQL** using **XiYan-SQLTraining** on **8×A100-80GB** (training), and deploy on **1×A100-80GB** for inference with **4 concurrent users at peak**.

Key principles:

- **Train:** LoRA on BF16 base weights (fast and stable on 8×80GB)
- **Package:** merge LoRA into the base model (single checkpoint)
- **Deploy:** quantize merged checkpoint to **4-bit** and serve via **vLLM** (paged attention) on 1×80GB
- **No overwrites:** do **not** modify XiYan-SQLTraining source files; add new files only.

---

## 2) Assumptions & Constraints

- Environment: **air-gapped / behind firewall**
- Training: **8× A100-80GB PCIe + NVLink**
- Inference: **1× A100-80GB PCIe**
- Peak concurrency: **4 concurrent generations**
- RAG prompt size: **6,500–16,000 characters** (≈ **1.6k–4k tokens**) + instructions + output
- Output contract: **Oracle SQL only** (no markdown, no explanations)
- Preferred serving: **vLLM** (for KV-cache efficiency at long context)

---

## 3) High-Level Blueprint

### Training (8×A100-80GB)
1. Build SFT dataset in XiYan “conversations” JSON format:
   - user: question + RAG schema context + rules
   - assistant: Oracle SQL only
2. Run **SFT LoRA (BF16)** with **Deepspeed ZeRO-3**.

### Packaging
3. Merge LoRA adapters into base weights → merged model.
4. Quantize merged model → **4-bit** (AWQ preferred, GPTQ fallback).

### Inference (1×A100-80GB)
5. Serve quantized model with **vLLM** configured for:
   - `max_model_len` (start at 8192)
   - `max_num_seqs=4`
   - output cap (256–512 tokens)
6. Add a **SQL guardrail + one retry**:
   - validate “SQL only”, block DML/DDL unless explicitly allowed
   - optional Oracle parse/EXPLAIN on a staging DB
   - retry once with error feedback if invalid

---

## 4) What we reuse from XiYan-SQLTraining

- `train/sft4xiyan.py` data masking pattern:
  - uses `tokenizer.apply_chat_template(...)`
  - masks prompt tokens with `IGNORE_TOKEN_ID`
- Deepspeed configs in `train/config/` (Zero-3)
- Adapter merge utility: `train/utils/adapter_merge.py`

---

## 5) What we avoid (for this Oracle-only project)

- Dialect routing / MoMQ / MOMQ-specific losses (keep `enable_dialect_router=false`)
- Multi-dialect training objectives (keep `sql_type` metadata only)

---

## 6) New Files to Add (No Overwrites)

Place all custom work under:

```
XiYan-SQLTraining/custom_oracle_llama/
```

### 6.1 Dataset builder
- `build_oracle_sft_dataset.py`

### 6.2 Training entrypoint (LoRA BF16)
- `sft_oracle_llama70b_lora.py`

### 6.3 Optional: 4-bit loader for experiments (QLoRA only)
- `train_util_4bit.py`
- `sft_oracle_llama70b_qlora.py` (optional)

### 6.4 Packaging: merge + quantize
- `package_oracle_model.py`

### 6.5 Inference: vLLM config + SQL guardrail
- `inference/vllm_config.yaml`
- `inference/sql_guardrail.py`

---

## 7) Required Prompt Format (must match production)

The dataset builder must create user messages like:

1. **Rules / contract** (SQL-only, Oracle dialect)
2. **RAG context** in structured sections:
   - Candidate tables
   - Join graph
   - Relevant columns
   - Business synonyms/mapping
   - Security filters (MOAC/RLS) (if applicable)
3. **Question**

---

## 8) Default Training Settings

- `model_max_length`: **8192**
- Precision: BF16
- LoRA target modules:
  - `q_proj,k_proj,v_proj,o_proj,up_proj,gate_proj,down_proj`
- Start LoRA params:
  - `r=32`, `alpha=32`, `dropout=0.05`
- Deepspeed ZeRO-3 config: `train/config/zero3.yaml` + `train/config/dp_zero3.json`

---

## 9) Packaging Requirements

- Merge adapters → HF-compatible merged model
- Quantize merged model → 4-bit using:
  - **AWQ** preferred, or **GPTQ** fallback
- Output a manifest JSON with:
  - base model identifier (path + hash if available)
  - adapter path + hash
  - dataset version id
  - quantization method + config

---

## 10) Inference Requirements (1×A100-80GB, 4 concurrent)

- vLLM with:
  - `max_model_len=8192` (tune down if needed)
  - `max_num_seqs=4`
  - `max_tokens` output cap 512 (SQL usually < 200)
- Enforce SQL-only and block unsafe statements
- One retry if invalid

---

## 11) Example Commands

### Build dataset
```bash
python custom_oracle_llama/build_oracle_sft_dataset.py \
  --input_raw data/oracle_raw.jsonl \
  --output_sft data/oracle_sft_conversations.json \
  --dataset_id oracle_ebs_v1
```

### Train (LoRA BF16)
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

### Merge + Quantize (AWQ)
```bash
python custom_oracle_llama/package_oracle_model.py \
  --base_model /models/llama-3.3-70b-instruct \
  --lora_adapter outputs/oracle_llama70b_lora \
  --merged_out outputs/merged_oracle_llama70b \
  --quant_out outputs/merged_oracle_llama70b_awq4 \
  --quant_method awq
```

### Serve (vLLM)
```bash
python -m vllm.entrypoints.openai.api_server \
  --model outputs/merged_oracle_llama70b_awq4 \
  --max-model-len 8192 \
  --max-num-seqs 4 \
  --max-tokens 512
```

---

## 12) Acceptance Criteria

- Training completes on 8×A100-80GB at `model_max_length>=8192` without OOM
- Merged model loads successfully
- Quantized 4-bit model loads on 1×A100-80GB
- vLLM serves **4 concurrent** requests with stable memory (no OOM) under configured caps
- On a staging Oracle schema eval set:
  - parse/EXPLAIN error rate under agreed threshold
  - hallucination rate (invalid column/table) under agreed threshold

