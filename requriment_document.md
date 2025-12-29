Below is a **developer-ready requirement document** to adopt the “8×A100-80GB LoRA fine-tune → merge → 4-bit quant → 1×A100-80GB inference (4 concurrent)” blueprint **using XiYan-SQLTraining**, while **only adding new files** (no overwriting existing framework files). Where we need changes (e.g., proper 4-bit loading), we’ll do it by adding **new scripts/modules** and calling them instead of modifying `train/trainer/train_util.py`.

---

# Requirement Document: Oracle EBS NL2SQL Fine-tuning & Deployment (XiYan-SQLTraining + LLaMA-3.1-70B-Instruct)

## 1) Goal

Fine-tune **LLaMA-3.1-70B-Instruct** for **Oracle EBS NL2SQL** using the XiYan-SQLTraining framework on **8×A100-80GB** (training), and deploy for inference on **1×A100-80GB** supporting **4 concurrent users at peak**, with **RAG prompt context ~6,500–16,000 chars**.

Key principles:

* **Training:** LoRA on BF16 base (fast & stable given 8×80GB)
* **Packaging:** merge LoRA adapters into base weights
* **Serving:** quantize merged model to **4-bit** and serve with an optimized engine (vLLM recommended)
* **Do not overwrite XiYan files** — add new files only.

---

## 2) Non-Goals (Explicit)

* No attempt to reproduce academic leaderboard results.
* No GRPO/RLHF phase in this first iteration.
* No multi-dialect routing / XiYan MOE dialect router for production (Oracle-only).

---

## 3) Inputs & Constraints

### Infrastructure

* Training: **8× A100-80GB PCIe NVLink**
* Inference: **1× A100-80GB PCIe**
* Environment: **air-gapped / behind firewall**, offline dependencies allowed only via internal artifact repo.

### Workload assumptions

* Peak concurrency: **4 concurrent generations**
* RAG prompt size: **6.5k–16k characters** (~1.6k–4k tokens, plus instructions & output)
* Output is **Oracle SQL only**, max output tokens typically **256–512**.

---

## 4) Architecture Overview

### Training Pipeline

1. Prepare Oracle EBS dataset in XiYan conversation format:

   * `conversations: [{role:user, content:...}, {role:assistant, content:...}]`
   * `assistant` content is **SQL only**
2. Run **SFT with LoRA** on `LLaMA-3.1-70B-Instruct` using Deepspeed ZeRO-3.
3. Save LoRA adapters and training artifacts.

### Packaging Pipeline

4. Merge LoRA adapter into base model (single merged checkpoint).
5. Quantize merged model to **4-bit** for inference deployment.

### Inference Pipeline

6. vLLM serving on 1×A100-80GB:

   * `max_model_len` set based on testing (target 8192 if feasible)
   * strict caps: output tokens, max concurrent sequences
7. Add **SQL validation loop** (parser/EXPLAIN) + one retry with error feedback.

---

## 5) XiYan-SQLTraining: What we reuse vs avoid

### Reuse (unchanged)

* Dataset formatting & masking logic in `train/sft4xiyan.py` pattern (it uses `tokenizer.apply_chat_template()` and masks prompt tokens)
* Deepspeed configs under `train/config/` (`zero3.yaml`, `dp_zero3.json`)
* Existing adapter merge script reference: `utils/adapter_merge.py` (framework mentions it)

### Avoid for this project

* XiYan multi-dialect router / MOE MOMQ paths
* “sql_type” router logic for training losses (Oracle-only; treat dialect as prompt text only)

---

## 6) Required Additions (New Files Only)

Create a new folder `custom_oracle_llama/` inside `XiYan-SQLTraining/` (or a sibling folder) that contains all customization code. The developer must **not modify** existing XiYan files.

### 6.1 New Training Entrypoint (LoRA, BF16 base)

**File:** `XiYan-SQLTraining/custom_oracle_llama/sft_oracle_llama70b_lora.py`

Purpose:

* A wrapper entrypoint that mirrors `train/sft4xiyan.py`, but:

  * sets Oracle-specific defaults
  * enforces SQL-only output in prompt conventions
  * uses the existing `trainer.train_util.load_tokenizer_and_model()` unchanged (LoRA only; no 4-bit loading required in training)

Must:

* accept the same CLI args as `train/sft4xiyan.py` (ModelArguments/TrainingArguments/DataArguments/LoraArguments)
* set:

  * `model_args.model_name_or_path` default to LLaMA-3.1-70B-Instruct path (internal)
  * `training_args.model_max_length` default to 8192 (override allowed)
  * `training_args.use_lora=True`
  * `lora_args.q_lora=False` (LoRA, not QLoRA)

### 6.2 New Optional QLoRA Loader (ONLY if we want QLoRA experiments)

You asked earlier about a tweak: `train/trainer/train_util.py` doesn’t truly load 4-bit; it only calls `prepare_model_for_kbit_training()`.

We will **not modify it**. Instead:

**File:** `XiYan-SQLTraining/custom_oracle_llama/train_util_4bit.py`

Purpose:

* Provide `load_tokenizer_and_model_4bit()` that performs:

  * `BitsAndBytesConfig(load_in_4bit=True, …)` and loads the base model quantized
  * then applies LoRA (QLoRA mode)
* This is optional for **experiments**; the main production plan uses LoRA BF16 training.

### 6.3 New Dataset Builder for Oracle EBS + RAG

**File:** `XiYan-SQLTraining/custom_oracle_llama/build_oracle_sft_dataset.py`

Purpose:

* Convert raw Oracle EBS NL2SQL training data into XiYan “conversations” JSON.
* Insert the RAG retrieved context into the user message in a deterministic structure.

#### Required input schema (minimum)

Each raw example must have:

* `question` (string)
* `sql_gold` (string, Oracle SQL)
* `rag_context` (string) — your retrieved schema/joins/aliases/synonyms
* optional:

  * `domain` (AP/PO/GL/…)
  * `org_policy` text / MOAC hints
  * `difficulty` tag

#### Output format (XiYan compatible)

A JSON array where each element contains:

* `conversations: [ {role:"user", content:"..."}, {role:"assistant", content:"..."} ]`
* `sql_type: "oracle"` (kept for metadata; not used for router)

#### Required prompt template (user content)

Must be stable and production-like, e.g.:

* Instruction header:

  * “Return **Oracle SQL only**. No markdown. No explanation.”
* RAG context sections (structured):

  * `# Candidate Tables`
  * `# Join Graph`
  * `# Relevant Columns`
  * `# Synonyms/Business Mapping`
  * `# Security Filters` (if applicable)
* Question at end.

### 6.4 New Packaging Script: Merge + Quantize

**File:** `XiYan-SQLTraining/custom_oracle_llama/package_oracle_model.py`

Purpose:

* Run:

  1. adapter merge (reuse XiYan merge utility or call it)
  2. quantization of merged model to 4-bit for inference

Quantization method requirement:

* Provide a pluggable option to choose one of:

  * AWQ (preferred for serving quality/latency)
  * GPTQ (fallback)
* Because environment is air-gapped, the quantization library must be installable offline.

Outputs:

* `merged_fp16_or_bf16_model/`
* `merged_quant4_model/`
* version manifest JSON:

  * base model hash/commit tag (internal)
  * adapter hash
  * dataset version
  * quantization config

### 6.5 New Inference Runbook + Config Templates

**File:** `XiYan-SQLTraining/custom_oracle_llama/inference/vllm_config.yaml`

Contains recommended defaults for:

* `max_model_len`: start 8192 (adjust after measurement)
* `max_num_seqs`: set for 4 concurrency
* `gpu_memory_utilization`: safe value (e.g., 0.90–0.93)
* `max_tokens` for generation: 256–512

**File:** `XiYan-SQLTraining/custom_oracle_llama/inference/sql_guardrail.py`

Purpose:

* A lightweight validation step that:

  * checks SQL is non-empty and Oracle-ish
  * blocks non-SELECT unless explicitly allowed
  * enforces org filters if your policy requires
  * (optional) runs `EXPLAIN PLAN`/parse step against staging DB
* Implements single retry:

  * if invalid, feed back error message and ask model to regenerate SQL only.

---

## 7) Training Configuration Requirements (8×A100)

### 7.1 Core training settings

* Fine-tuning method: **LoRA (BF16 base)**
* Deepspeed: **ZeRO Stage 3**
* Gradient checkpointing: ON (recommended)
* FlashAttention: ON if compatible with your HF stack
* Max sequence length: start **8192**

### 7.2 LoRA settings (baseline)

* target modules: `q_proj,k_proj,v_proj,o_proj,up_proj,gate_proj,down_proj` (good default for LLaMA)
* `r`: 16–64 (start 32)
* `alpha`: 16–64 (start 32)
* dropout: 0.05

### 7.3 Data settings

* Ensure consistent “SQL only” targets
* Add negative examples only if you have a validator loop; otherwise keep training clean.

### 7.4 Checkpointing

* Save LoRA adapter weights frequently (e.g., every N steps)
* Save training config + git SHA of framework + dataset version

---

## 8) Packaging Requirements (Merge + Quantize)

### 8.1 Merge

* Must produce a single HF-compatible merged model
* Must record provenance in a manifest

### 8.2 Quantize

* Quantize merged weights to **4-bit** for a single A100-80GB
* Validate:

  * model loads and generates
  * SQL-only compliance
  * no major regression on Oracle parse success

---

## 9) Inference Requirements (1×A100-80GB, 4 concurrent)

### 9.1 Serving engine

* Preferred: **vLLM** (paged attention helps concurrency with long context)

### 9.2 Limits & policies (must implement)

* Strict output limit: **max 512 tokens**
* Max concurrent sequences: sized for **4**
* Enforce “SQL only” with post-check + 1 retry
* Timeouts (generation cap and request timeout)

### 9.3 Validation metrics

Track:

* parse/EXPLAIN failure rate
* retry rate
* p50/p95 latency
* tokens/sec
* hallucinated column/table rate

---

## 10) Acceptance Criteria

### Training

* LoRA training completes on 8×A100-80GB without OOM at `model_max_length >= 8192`
* Produces adapter checkpoints + logs + manifest

### Packaging

* Adapter merges successfully into base model
* Quantized 4-bit model fits on **1×A100-80GB**
* Quantized model can run at **4 concurrent** within set latency target (define internally)

### Quality (minimum)

On a staging Oracle schema eval set:

* ≥ X% SQL parses successfully
* ≥ Y% execution accuracy (or proxy metric if execution unavailable)
* hallucination rate below threshold

(Developer should implement the metric harness; thresholds set by your team.)

---

## 11) Developer Task List (Implementation Plan)

### Phase 1 — Add new scripts (no edits)

1. Add `custom_oracle_llama/` directory structure
2. Implement dataset builder `build_oracle_sft_dataset.py`
3. Implement training wrapper `sft_oracle_llama70b_lora.py`
4. Implement packaging script `package_oracle_model.py`
5. Implement vLLM config template + SQL guardrail

### Phase 2 — Runbook & CI-like checks

6. Add a “smoke test”:

   * render one prompt via `apply_chat_template()`
   * ensure masking boundary works (assistant-only loss)
7. Add a “load test” script for 4 concurrent on a single GPU

### Phase 3 — Optional QLoRA experimentation

8. Implement `train_util_4bit.py` and optional `sft_oracle_llama70b_qlora.py`

---

## 12) Commands / Runbook (Examples)

### Build dataset

```bash
python custom_oracle_llama/build_oracle_sft_dataset.py \
  --input_raw data/oracle_raw.jsonl \
  --output_sft data/oracle_sft_conversations.json
```

### Train (LoRA BF16, 8×GPU)

```bash
accelerate launch --config_file train/config/zero3.yaml \
  custom_oracle_llama/sft_oracle_llama70b_lora.py \
  --model_name_or_path /models/llama-3.1-70b-instruct \
  --data_path data/oracle_sft_conversations.json \
  --output_dir outputs/oracle_llama70b_lora \
  --model_max_length 8192 \
  --use_lora True \
  --q_lora False
```

### Merge + Quantize

```bash
python custom_oracle_llama/package_oracle_model.py \
  --base_model /models/llama-3.1-70b-instruct \
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
  --max-num-seqs 4
```

---

## 13) What I need from you (to finalize the doc into a “handoff-ready” PR checklist)

No more code is needed from you, but to lock down the runbook values:

* your internal path/naming convention for the LLaMA base model folder
* whether you will use AWQ or GPTQ internally
* whether you have a staging Oracle instance available for EXPLAIN/parse validation

If you don’t answer, the developer can still proceed with AWQ-first and a parser-only guardrail.

---

If you want, I can also generate the **exact folder tree** + a **PR checklist** (review steps, tests, and deployment gating) formatted exactly the way your engineering team likes (Jira-ready / Notion-ready / markdown).
