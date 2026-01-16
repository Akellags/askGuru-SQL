Absolutely — here’s a **complete, developer-handoff-ready** package in **Markdown + full working code** for every new file I proposed, designed to work with **askGuru-SQL** while **not overwriting any existing code** (everything is additive).

I also bundled everything into a zip so your dev can drop it into the repo quickly:

**[Download: oracle_askguru_llama_requirements_and_code.zip](sandbox:/mnt/data/oracle_askguru_llama_requirements_and_code.zip)**

---

## What’s included in the zip

### ✅ Requirements Document

* `REQUIREMENTS.md` — the full markdown requirement doc (training → merge → quantize → serve, 4 concurrency)

### ✅ New additive code (no overwrites)

Folder: `custom_oracle_llama/`

* `README.md`
* `build_oracle_sft_dataset.py`
* `sft_oracle_llama70b_lora.py`
* `package_oracle_model.py`
* `train_util_4bit.py` *(optional, for QLoRA experiments)*
* `sft_oracle_llama70b_qlora.py` *(optional, for QLoRA experiments)*
* `inference/vllm_config.yaml`
* `inference/sql_guardrail.py`

---

## How your developer should use this (quick start)

### 1) Copy into your askGuru-SQL repo root

Place the folder **inside** `askGuru-SQL/`:

```
askGuru-SQL/
  custom_oracle_llama/
  train/
  evaluation/
  data/
  ...
```

### 2) Build dataset (Oracle EBS → askGuru conversations JSON)

```bash
python custom_oracle_llama/build_oracle_sft_dataset.py \
  --input_raw data/oracle_raw.jsonl \
  --output_sft data/oracle_sft_conversations.json \
  --dataset_id oracle_ebs_v1
```

### 3) Train (LoRA BF16 on 8×A100-80GB)

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

### 4) Merge + quantize to 4-bit (for 1×A100-80GB inference)

```bash
python custom_oracle_llama/package_oracle_model.py \
  --base_model /models/llama-3.3-70b-instruct \
  --lora_adapter outputs/oracle_llama70b_lora \
  --merged_out outputs/merged_oracle_llama70b \
  --quant_out outputs/merged_oracle_llama70b_awq4 \
  --quant_method awq
```

### 5) Serve on 1×A100-80GB (4 concurrent)

```bash
python -m vllm.entrypoints.openai.api_server \
  --model outputs/merged_oracle_llama70b_awq4 \
  --max-model-len 8192 \
  --max-num-seqs 4 \
  --max-tokens 512
```

Then wrap responses with `custom_oracle_llama/inference/sql_guardrail.py` to enforce **SQL-only + 1 retry**.

---

## Notes on “no overwrites”

You asked specifically to avoid changing askGuru files like `train/trainer/train_util.py`. This approach:

* **does not modify askGuru code**
* **reuses askGuru trainer + collator + merge utility**
* implements “tweaks” only by **new scripts** (for example `train_util_4bit.py` if you want real QLoRA later)

---

If you want, next I can also add (as additive files):

* a small `smoke_test_prompt_template.py` that prints `apply_chat_template()` output + verifies masking boundaries
* a `load_test_vllm_4concurrent.py` to sanity-check 4 concurrent users on the single A100 before production rollout
