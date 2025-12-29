# custom_oracle_llama

This folder contains **additive** scripts to adapt XiYan-SQLTraining for:

- Oracle EBS NL2SQL
- LLaMA-3.1-70B-Instruct
- 8×A100-80GB training (LoRA BF16)
- 1×A100-80GB inference (4-bit quant, 4 concurrent)

No existing XiYan-SQLTraining files are modified.

## Files

- `build_oracle_sft_dataset.py`  
  Converts raw JSON/JSONL into XiYan `conversations` SFT format.

- `sft_oracle_llama70b_lora.py`  
  Training entrypoint (LoRA BF16) reusing XiYan trainer + collator.

- `train_util_4bit.py` and `sft_oracle_llama70b_qlora.py` (optional)  
  True 4-bit loading utilities for QLoRA experiments.

- `package_oracle_model.py`  
  Merge LoRA -> merged model, then quantize merged model (AWQ/GPTQ) and write manifest.

- `inference/vllm_config.yaml`  
  Suggested vLLM settings (start values) for 1×A100-80GB.

- `inference/sql_guardrail.py`  
  SQL-only validation + one-retry prompt helper.

## Typical flow

1. Build dataset
2. Train LoRA
3. Merge adapter -> merged model
4. Quantize merged model -> 4-bit
5. Serve with vLLM + guardrail loop
