# Dual-Model Strategy: LLaMA-3.1-70B + SQLCoder-70B

**Framework for deploying and managing two complementary models for Oracle EBS NL2SQL**

---

## Executive Summary

You now have **two separate codebases** for fine-tuning and inference:

| Model | Codebase | Best For | Accuracy | Context | Status |
|-------|----------|----------|----------|---------|--------|
| **LLaMA-3.1-70B** | `custom_oracle_llama/` | Complex multi-table joins, business logic | ~52-60% SQL | 128K | ✅ PRIMARY |
| **SQLCoder-70B** | `custom_oracle_sqlcoder/` | Pure SQL, simple queries, date/GROUP BY | 93% SQL | 16K | 🔄 SECONDARY |

---

## Directory Structure

```
askGuru-SQL/
├── custom_oracle_llama/           (PRIMARY)
│   ├── _preprocessing_utils.py    (LLaMA chat template)
│   ├── sft_oracle_llama70b_lora.py
│   ├── inference/
│   │   ├── sql_guardrail.py
│   │   └── vllm_config.yaml
│   └── README.md
│
├── custom_oracle_sqlcoder/        (SECONDARY)
│   ├── _preprocessing_sqlcoder.py (SQLCoder prompt format)
│   ├── sft_oracle_sqlcoder70b_lora.py
│   ├── inference_oracle_sqlcoder.py
│   ├── sqlcoder_join_validator.py (POST-PROCESSING)
│   └── README.md
│
├── data/oracle_sft_conversations/ (SHARED)
│   ├── oracle_sft_conversations_train.json
│   ├── oracle_sft_conversations_val.json
│   ├── oracle_sft_conversations_test.json
│   └── oracle_sft_conversations_full.json
│
├── DEPLOY_FINETUNE_UBUNTU24.md    (LLaMA training)
├── DEPLOY_INFERENCE_UBUNTU24.md   (LLaMA inference)
├── DEPLOY_SQLCODER_UBUNTU24.md    (SQLCoder training & inference)
└── DEPLOYMENT_GUIDE_SUMMARY.md
```

---

## Training Timeline

Both models train on **the same dataset** (4,822 examples). You can run sequentially or in parallel on separate hardware.

### Sequential (Single 8×A100)

```
Week 1:
├─ Day 1-2: Train LLaMA-3.1-70B (10 hours)
├─ Day 3-4: Train SQLCoder-70B (10 hours)
└─ Day 5: Evaluation & comparison

Total: ~5-6 days
```

### Parallel (Two 8×A100 clusters)

```
Week 1:
├─ Cluster A: LLaMA-3.1-70B (10 hours)
├─ Cluster B: SQLCoder-70B (10 hours)
└─ Overlap: Day 1-2, evaluate results Day 3

Total: ~3 days
```

---

## Deployment Options

### Option 1: Primary Only (Recommended for MVP)

Deploy **LLaMA-3.1-70B only**:
- Single model endpoint
- Simpler operations
- ~52-60% accuracy on Oracle SQL
- Sufficient for most use cases after fine-tuning

**When to choose**: If you want simplicity, risk tolerance is high, fine-tuned model performs well.

**Deployment**:
```bash
# Use existing deployment guides
./deploy_finetune.sh
./deploy_inference.sh
```

### Option 2: Dual Model with Sequential Fallback

Deploy **both models** with routing:
1. First attempt: LLaMA-3.1-70B (primary, general reasoning)
2. If confidence < threshold, try SQLCoder-70B (specialist)
3. Validate using `sqlcoder_join_validator.py`

**When to choose**: If you want safety net, have resources for 2 models.

**Deployment**:
```python
# Pseudo-code (implement in your app)
def generate_sql(question: str, schema: str) -> str:
    # Try primary
    llama_sql, llama_confidence = llama_model.generate(question, schema)
    
    if llama_confidence < 0.7:
        # Try secondary
        sqlcoder_sql = sqlcoder_model.generate(question, schema)
        # Validate joins
        if sqlcoder_join_validator.validate(sqlcoder_sql, schema):
            return sqlcoder_sql
    
    return llama_sql
```

### Option 3: Query-Specific Routing

Route queries based on characteristics:
- **Simple queries** (1-2 tables) → SQLCoder (fast, accurate for simple SQL)
- **Complex queries** (3+ tables, window functions) → LLaMA (better reasoning)

**When to choose**: If you have good query classification, want to optimize latency.

**Routing logic**:
```python
def route_query(question: str, num_tables: int) -> str:
    if num_tables <= 2 and len(question) < 100:
        return "sqlcoder"  # Fast, accurate for simple cases
    else:
        return "llama"     # Better reasoning for complex logic
```

### Option 4: Ensemble Voting

Generate with both, compare outputs:
1. LLaMA generates SQL
2. SQLCoder generates SQL
3. If outputs match → high confidence, use either
4. If outputs differ → apply scoring heuristic

**When to choose**: If you need highest accuracy, have time for multiple generations.

**Scoring**:
```python
def ensemble_vote(llama_sql: str, sqlcoder_sql: str, schema: str) -> str:
    if normalize_sql(llama_sql) == normalize_sql(sqlcoder_sql):
        return llama_sql  # Consensus → high confidence
    
    # Different outputs
    llama_score = llama_confidence(llama_sql, schema)
    sqlcoder_score = sqlcoder_confidence(sqlcoder_sql, schema)
    
    # Validation check (SQLCoder's weak point)
    sqlcoder_join_valid, _ = validate_sql_joins(sqlcoder_sql, schema)
    if not sqlcoder_join_valid:
        sqlcoder_score *= 0.7  # Penalize bad joins
    
    return llama_sql if llama_score > sqlcoder_score else sqlcoder_sql
```

---

## Training Commands

### LLaMA-3.1-70B

```bash
accelerate launch --config_file train/config/zero3.yaml \
  custom_oracle_llama/sft_oracle_llama70b_lora.py \
  --model_name_or_path meta-llama/Llama-3.1-70b-instruct \
  --data_path data/oracle_sft_conversations/oracle_sft_conversations_train.json \
  --output_dir outputs/oracle_llama70b_lora \
  --model_max_length 8192 \
  --use_lora True \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2.0e-4 \
  --num_train_epochs 3
```

### SQLCoder-70B

```bash
accelerate launch --config_file train/config/zero3.yaml \
  custom_oracle_sqlcoder/sft_oracle_sqlcoder70b_lora.py \
  --model_name_or_path defog/sqlcoder-70b-alpha \
  --data_path data/oracle_sft_conversations/oracle_sft_conversations_train.json \
  --output_dir outputs/oracle_sqlcoder70b_lora \
  --model_max_length 4096 \
  --use_lora True \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2.0e-4 \
  --num_train_epochs 3
```

---

## Inference Comparison

### LLaMA Inference

```python
from custom_oracle_llama.sft_oracle_llama70b_lora import LLaMAInference

llama = LLaMAInference("outputs/oracle_llama70b_lora")

sql = llama.generate(
    question="List suppliers with invoice amount > 100K",
    schema=schema,
    max_tokens=512
)
```

### SQLCoder Inference

```python
from custom_oracle_sqlcoder.inference_oracle_sqlcoder import SQLCoderInference

sqlcoder = SQLCoderInference("outputs/oracle_sqlcoder70b_lora")

sql = sqlcoder.generate(
    question="List suppliers with invoice amount > 100K",
    schema=schema,
    validate_joins=True  # Post-processing validation
)
```

---

## Decision Matrix: Which to Deploy?

| Requirement | LLaMA Only | Both Models |
|---|---|---|
| **Resource-constrained** | ✅ | ❌ |
| **Highest accuracy needed** | ❌ | ✅ |
| **Complex joins common** | ✅ | ✅ |
| **Simple queries only** | ⚠️ | ✅ |
| **Fast inference critical** | ✅ | ❌ |
| **Easy operations** | ✅ | ❌ |
| **Safety net needed** | ❌ | ✅ |
| **MVP timeline** | ✅ | ❌ |

### Recommendation

**Start with LLaMA-only MVP:**
1. Train & deploy LLaMA-3.1-70B
2. Evaluate on test set (483 examples)
3. Measure accuracy, latency, cost
4. If accuracy < 70%, add SQLCoder as backup
5. If performing well, stick with single model for ops simplicity

---

## Evaluation Protocol

### Test Set Metrics

For each model, measure:

```python
from datasets import load_dataset
import json

test_data = load_dataset("json", data_files="data/oracle_sft_conversations/oracle_sft_conversations_test.json")

# For each example:
correct = 0
for example in test_data["train"]:
    question = example["conversations"][0]["content"]
    expected_sql = example["conversations"][1]["content"]
    
    # Generate
    generated_sql = model.generate(question, schema)
    
    # Evaluate
    if sql_match(normalize(generated_sql), normalize(expected_sql)):
        correct += 1

accuracy = correct / len(test_data["train"])
print(f"Accuracy: {accuracy:.1%}")
```

### Comparison Template

```
═══════════════════════════════════════════════════════════
Test Set Evaluation (483 examples)
═══════════════════════════════════════════════════════════

LLaMA-3.1-70B:
  ✓ Overall Accuracy: XX%
  ✓ Correct: XXX / 483
  ✓ Avg Latency: XX ms
  ✓ Memory: XX GB
  ✓ Strengths: [...]
  ✗ Weaknesses: [...]

SQLCoder-70B:
  ✓ Overall Accuracy: XX%
  ✓ Correct: XXX / 483
  ✓ Avg Latency: XX ms
  ✓ Memory: XX GB
  ✓ Strengths: [...]
  ✗ Weaknesses: [...]

Decision: [LLaMA Only / Both / SQLCoder Only]
Reasoning: [...]
═══════════════════════════════════════════════════════════
```

---

## Monitoring in Production

If deploying both models, monitor:

```python
# Track requests routed to each model
router_metrics = {
    "llama_requests": 0,
    "sqlcoder_requests": 0,
    "llama_success": 0,
    "sqlcoder_success": 0,
    "routing_time_ms": 0
}

# Per-request logging
{
    "timestamp": "2025-01-15T10:30:00Z",
    "question": "List suppliers...",
    "routed_to": "llama",
    "confidence": 0.87,
    "sql": "SELECT ...",
    "validation_passed": true,
    "latency_ms": 245
}
```

---

## Migration Path: Single → Dual

If you deploy LLaMA only now, migrating to dual model later:

1. **Keep dataset & preprocessing identical**
   - Both models train on `data/oracle_sft_conversations/`
   - No data re-engineering needed

2. **Train SQLCoder independently**
   - Run on separate hardware or after LLaMA completes
   - No changes to LLaMA production setup

3. **Add routing layer gradually**
   - Start 100% LLaMA
   - A/B test 10% SQLCoder queries
   - Increase SQLCoder traffic if successful
   - Implement fallback logic

4. **Monitoring transition**
   - Day 1-14: LLaMA only (baseline)
   - Day 15-28: LLaMA + SQLCoder on 10% (pilot)
   - Day 29-42: LLaMA + SQLCoder on 50% (ramp up)
   - Day 43+: Full dual model with confidence-based routing

---

## Cost Comparison (Monthly, AWS pricing)

### LLaMA-Only
- Fine-tuning: 8×A100 × 10 hours = $1,200
- Inference: 1×A100 × 730 hours = $2,190
- **Total**: ~$3,400/month

### Dual Model
- Fine-tuning: 8×A100 × 20 hours = $2,400 (train both)
- Inference: 2×A100 × 730 hours = $4,380 (both running)
- **Total**: ~$6,780/month

**Savings with LLaMA-only: 50%**

---

## Next Steps

1. **Decision**: Choose deployment option (LLaMA-only recommended for MVP)
2. **Train**: Execute training scripts
3. **Evaluate**: Test on 483 example test set
4. **Deploy**: Use provided deployment guides
5. **Monitor**: Track accuracy, latency, cost
6. **Iterate**: Add SQLCoder if needed

---

## References

- LLaMA Codebase: `custom_oracle_llama/README.md`
- SQLCoder Codebase: `custom_oracle_sqlcoder/README.md`
- LLaMA Deployment: `DEPLOY_FINETUNE_UBUNTU24.md`, `DEPLOY_INFERENCE_UBUNTU24.md`
- SQLCoder Deployment: `DEPLOY_SQLCODER_UBUNTU24.md`
