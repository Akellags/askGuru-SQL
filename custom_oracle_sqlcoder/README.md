# Oracle EBS NL2SQL: SQLCoder-70B Secondary Model

**SQLCoder-70B** as secondary backup model for Oracle EBS natural language to SQL conversion.

> **⚠️ Status**: Secondary model only. Use **LLaMA-3.3-70B** as primary (or 3.1).
>
> Reasons:
> - SQLCoder has 93% overall accuracy vs LLaMA's ~52%
> - But 85.7% JOIN accuracy (Oracle EBS is JOIN-heavy)
> - 16K context window (LLaMA has 128K)
> - Use SQLCoder only if LLaMA fails or for simple queries

## Overview

- **Base Model**: CodeLlama-70B (fine-tuned by Defog on SQL data)
- **License**: Apache 2.0
- **Model ID**: `defog/sqlcoder-70b-alpha`
- **Architecture**: LlamaForCausalLM (compatible with askGuru framework)
- **SQL Accuracy**: 93% on unseen schemas
- **Training**: 8×A100-80GB LoRA BF16 + DeepSpeed ZeRO-3
- **Inference**: 1×A100-80GB with 4-bit quantization

## Files

### Training & Preprocessing

- **`_sqlcoder_utils.py`**: Shared utilities for prompt formatting, SQL cleaning, and validation.
- **`_preprocessing_sqlcoder.py`**: SQLCoder-specific dataset preprocessing for training.
- **`build_oracle_sft_dataset_sqlcoder.py`**: Separate dataset generation script that outputs SQLCoder-native prompt formats.
- **`sft_oracle_sqlcoder70b_lora.py`**: LoRA training entrypoint.

### Inference & Validation

- **`inference_oracle_sqlcoder.py`**: High-level inference engine with integrated cleaning.
- **`sqlcoder_join_validator.py`**: Post-processing validator for JOIN correctness.

## Quick Start

### 1. Build SQLCoder-specific Dataset

Unlike the Llama model, SQLCoder performs best with its native `### Task` prompt format. We create a separate dataset to avoid mixing structures:

```bash
python custom_oracle_sqlcoder/build_oracle_sft_dataset_sqlcoder.py \
  --config data/oracle_sft_config.yaml
```

This will produce:
- `data/oracle_sft_conversations/oracle_sqlcoder_sft_train.json`
- `data/oracle_sft_conversations/oracle_sqlcoder_sft_val.json`

### 2. Fine-tuning

```bash
accelerate launch --config_file train/config/zero3.yaml \
  custom_oracle_sqlcoder/sft_oracle_sqlcoder70b_lora.py \
  --model_name_or_path defog/sqlcoder-70b-alpha \
  --data_path data/oracle_sft_conversations/oracle_sqlcoder_sft_train.json \
  --eval_data_path data/oracle_sft_conversations/oracle_sqlcoder_sft_val.json \
  --output_dir outputs/oracle_sqlcoder70b_lora \
  --model_max_length 4096 \
  --use_lora True \
  --q_lora False
```

**Key differences from LLaMA:**
- Model: `defog/sqlcoder-70b-alpha` (not Meta's LLaMA)
- `model_max_length`: 4096 (16K context available, but 4K safe for safety margin)
- No dialect router (SQL-only)

**Training time**: ~8-10 hours (3 epochs, 4,822 examples)

**Memory per GPU**: ~50-55GB (similar to LLaMA)

### 3. Inference

**Standalone:**
```bash
python custom_oracle_sqlcoder/inference_oracle_sqlcoder.py \
  --model_path outputs/oracle_sqlcoder70b_lora \
  --question "List active suppliers" \
  --schema_file schemas/oracle_ebs.txt \
  --validate_joins
```

**As module:**
```python
from custom_oracle_sqlcoder.inference_oracle_sqlcoder import SQLCoderInference

inference = SQLCoderInference("outputs/oracle_sqlcoder70b_lora")

sql = inference.generate(
    question="List suppliers by ledger",
    schema=schema_text,
    validate_joins=True
)

print(sql)
```

**Batch inference:**
```python
questions = ["Q1", "Q2", "Q3"]
schemas = [schema1, schema2, schema3]

sqls = inference.generate_batch(questions, schemas)
```

## SQLCoder Prompt Format

SQLCoder expects a different prompt than LLaMA:

```
### Task
Generate a SQL query to answer the following question: `{question}`

### Database Schema
{schema}

### SQL Query
{sql}
```

Preprocessing automatically handles conversion from askGuru format.

## Benchmark Comparison

| Metric | SQLCoder-70B | LLaMA-3.3-70B | Winner |
|--------|---|---|---|
| Overall Accuracy | 93% | ~55-65% | ✅ SQLCoder |
| GROUP BY | 96% | ? | ✅ SQLCoder |
| WHERE | 97.1% | ? | ✅ SQLCoder |
| **JOIN** | **85.7%** | ? | ⚠️ Weak |
| ORDER BY | 91.4% | ? | ✅ SQLCoder |
| Context Window | 16K | 128K | ✅ LLaMA |
| General Reasoning | Unknown | Superior | ✅ LLaMA |

**Verdict**: SQLCoder wins on pure SQL generation but **loses on complex JOINs**. For Oracle EBS (multi-table queries), LLaMA's general reasoning + fine-tuning on your data is safer.

## When to Use SQLCoder

**Use SQLCoder-70B if:**
- Query is simple (single table or 1-2 simple joins)
- Heavy focus on date handling, WHERE clauses, GROUP BY
- Primary LLaMA model is struggling with SQL syntax
- You want a SQL-specialized second opinion

**Avoid SQLCoder-70B if:**
- Query requires 3+ table JOINs
- Complex business logic needed
- Context exceeds 8K tokens (half of 16K limit)

## Known Issues & Limitations

### 1. JOIN Accuracy (85.7%)

**Problem**: SQLCoder struggles with complex multi-table JOINs.

**Solution**: `sqlcoder_join_validator.py` catches these. Use `validate_joins=True` in inference.

Example error caught:
```sql
SELECT * FROM suppliers s WHERE s.vendor_id = p.vendor_id
-- Missing JOIN keyword → caught & flagged
```

### 2. Context Window (16K)

**Problem**: Oracle EBS schemas can exceed 8K tokens when comprehensive.

**Solution**:
- Limit schema to relevant tables only
- Summarize large schemas
- Use vector retrieval for table selection
- Max `model_max_length=4096` in training for safety

### 3. Alpha Status

**Problem**: `defog/sqlcoder-70b-alpha` is not officially released.

**Solution**:
- Monitor [Defog GitHub](https://github.com/defog-ai/sqlcoder) for updates
- Check for newer versions with larger context window
- If available, update training script

## Deployment Strategy

### Option A: Sequential (Recommended)

1. Run query through **LLaMA-3.3-70B** (primary)
2. If LLaMA confidence low, try **SQLCoder-70B** (secondary)
3. Validate results with `sqlcoder_join_validator.py`

### Option B: Ensemble

1. Generate SQL with both models
2. Compare outputs
3. Use voting or confidence scores to select best

### Option C: Fallback Only

- Deploy only LLaMA in production
- SQLCoder trained but unused until needed

## Production Deployment

See `DEPLOY_SQLCODER_UBUNTU24.md` for:
- Dependency installation
- Model download & setup
- LoRA merge & quantization
- vLLM server deployment
- Health checks & monitoring

## Reusing Training Dataset

The dataset from `custom_oracle_llama/` is **directly reusable**:

```bash
# Dataset location (created by build_oracle_sft_dataset.py)
data/oracle_sft_conversations/
├── oracle_sft_conversations_train.json    (3,857 examples)
├── oracle_sft_conversations_val.json      (482 examples)
├── oracle_sft_conversations_test.json     (483 examples)
└── oracle_sft_conversations_full.json     (4,822 examples)

# Both models share this data:
# - LLaMA uses: askGuru chat template format (apply_chat_template)
# - SQLCoder uses: SQLCoder prompt format (custom converter)
# - Both trained on identical examples, just different prompts
```

## Troubleshooting

### OOM Errors

Reduce batch size:
```bash
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 16  # Effective batch: 2*16*8=256
```

### Poor SQL Quality

1. Check model downloaded correctly: `defog/sqlcoder-70b-alpha`
2. Verify schema is properly formatted
3. Use `--validate_joins` to catch errors
4. Check if dataset splits are balanced

### Context Window Exceeded

Reduce `model_max_length`:
```bash
--model_max_length 2048  # Conservative limit
```

### Model Not Found

Ensure HuggingFace access:
```bash
huggingface-cli login
huggingface-cli whoami
```

## References

- [Defog GitHub](https://github.com/defog-ai/sqlcoder)
- [SQLCoder HuggingFace](https://huggingface.co/defog/sqlcoder-70b-alpha)
- [SQL-Eval Benchmark](https://github.com/defog-ai/sql-eval)

## Next Steps

1. Copy training commands to `CLAUDE.md` for easy reuse
2. Decide: Deploy both models or LLaMA-only?
3. If deploying both, set up routing logic (simple queries → SQLCoder, complex → LLaMA)
4. Monitor Defog releases for context window improvements
