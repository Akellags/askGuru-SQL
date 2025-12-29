# Complete End-to-End Guide: NL-to-SQL with Critic Loop

## Overview: Full Framework Capabilities Beyond Fine-tuning

The askGuru-SQL framework is **much more than just a training tool**. It's a complete **NL-to-SQL production system** with:

1. **Data Processing**: Raw data → M-Schema representation
2. **Data Augmentation**: Schema shuffling, filtering, permutation
3. **Fine-tuning**: Full training pipeline with multi-dialect support
4. **Inference**: Load finetuned models and generate SQL
5. **Self-Refinement**: Critic loop with execution feedback
6. **Evaluation**: Multi-metric assessment (exact match, execution match)
7. **Production Ready**: Support for PostgreSQL, MySQL, SQLite, Cypher, nGQL

---

## 🎯 Complete Production Workflow

### Step 1: Setup & Model Loading

```bash
# Download and prepare model
cd train/utils
python model_download.py

# Or use your finetuned model
ls -la /path/to/finetuned/model/
```

### Step 2: Prepare Test Data with Schemas

Create `evaluation/test_data.json`:
```json
[
  {
    "id": 0,
    "db_name": "movie_db",
    "db_id": "movie_db",
    "question": "Find top 10 highest-rated movies from 2020 onwards",
    "conversations": [
      {
        "role": "user",
        "content": "You are a SQLite expert。database schema：...\n[User Question]\nFind top 10 highest-rated movies from 2020 onwards..."
      }
    ]
  }
]
```

### Step 3: Run Inference

```bash
cd evaluation
python sql_infer.py \
  --model_name_or_path /path/to/finetuned/model \
  --lora_path ""  # If using LoRA adapter \
  --test_set_path test_data.json \
  --expr_version mistral_prod_v1 \
  --batch_size 8 \
  --use_flash_attention True
```

**Output**: `output/mistral_prod_v1/mistral_prod_v1_YYYYMMDD_results.json`

---

## 🔄 Critic Loop: Self-Refinement Pipeline

### Architecture

```
Question + Schema
      ↓
[Model] Generate SQL
      ↓
[Execute] Try on Database
      ↓
      ├─→ Success ✓ → Output SQL
      │
      └─→ Error ✗ → Capture Error
                    ↓
                  [Self-Refine Model]
                  Input: Question + Error + Schema
                    ↓
                  [Execute] Try Again
                    ↓
                    ├─→ Success ✓ → Output
                    └─→ Still Error → Attempt #3
```

This comprehensive guide covers the complete end-to-end workflow for NL-to-SQL generation with critic loops.