# askGuru-SQL: Complete Migration & Learning Context

## 📍 Project Status

**Migration Date**: December 26, 2025  
**Source Framework**: askGuru-SQL (Apache 2.0 Licensed)  
**New Location**: `c:\Users\ALIENWARE\Projects\askGuru-SQL`  
**Framework Type**: Finetuning Mistral-Small-3.2-24B for NL-to-SQL generation  
**Target Model**: Mistral-Small-3.2-24B-Instruct-2506

---

## 🎯 What You Have (Complete Breakdown)

### 1. **Core Framework Architecture**

#### Training Pipeline (`train/`)
- **Main Entry**: `sft4askguru.py` - SFT (Supervised Fine-Tuning) training script
- **Trainer Classes** (`train/trainer/`):
  - `DeepCustomTrainer`: Custom trainer with LoRA, distributed training support
  - `argument.py`: Configuration for models, data, LoRA, training hyperparameters
  - `train_util.py`: Tokenizer loading, model initialization, data collator
  - `moe_momq.py`: Mixture of Experts support (optional)

- **Training Scripts**:
  - `askguru_sft.sh`: Standard training script
  - `askguru_momq_sft.sh`: MoE training variant
  - `train/config/`: DeepSpeed ZeRO configs (zero1, zero2, zero2_offload, zero3)

#### Data Processing (`data/`)
- **Data Pipeline**:
  - `data_processing.py`: Raw data → processed with M-Schema
  - `data_assembler.py`: Processed data → training format (conversations)
  
- **Key Utilities** (`data/data_utils/`):
  - `m_schema.py`: M-Schema generation (intelligent schema representation)
  - `prompt_utils.py`: Prompt templates for NL2SQLite, NL2PostgreSQL, self-refinement, SQL selection
  - `schema_engine.py`: Database schema extraction
  - `type_engine.py`: Type conversions
  - `aug_ops/`: Data augmentation (SchemaShuffle, SchemaFilter, SchemaPermute, SQLTranslate)

#### Evaluation (`evaluation/`)
- **Inference Scripts**:
  - `sql_infer.py`: Single candidate generation
  - `sql_multi_candidate_infer.py`: **NEW** - Multiple SQL generation + critic selection
  
- **Evaluation**:
  - `sql_eval.py`: Execution-based evaluation
  - `eval_utils/`: SQL utilities, value matching, metrics calculation

---

### 2. **Key Technologies & Techniques**

#### **LoRA (Parameter-Efficient Finetuning)**
- **Config**: R=512, alpha=512 (for 24B models)
- **Target Modules**: q_proj, k_proj, v_proj, o_proj, up_proj, gate_proj, down_proj
- **Benefit**: Reduces training memory by 90%, faster training, deployable adapters

#### **Multi-Dialect Support**
- SQLite, PostgreSQL, MySQL, Cypher, nGQL
- Easy to add Oracle EBS (see `oracle_ebs_implementation.py`)
- Template-based prompt system for each dialect

#### **Self-Refinement (Critic Loop)**
- Generate SQL → Execute → If error, refine using error message
- Template: `SQLITE_SELF_REFINE_TEMPLATE` in `prompt_utils.py`
- Max 2-3 attempts, then fallback to candidate selection

#### **Multi-Candidate SQL Generation & Critic Selection**
Three methods in `sql_multi_candidate_infer.py`:
1. **Beam Search** (deterministic): 5 beam candidates
2. **Temperature Sampling** (diverse): 5 temperature-sampled candidates  
3. **Hybrid** (recommended): 2-3 beam + 2-3 sampled candidates

Critic model then selects best candidate using `SQL2SELECT_TEMPLATE`:
- Compares execution results
- Analyzes correctness
- Selects optimal SQL

#### **Data Augmentation**
- **SchemaShuffle**: Randomize table/column order (50% prob)
- **SchemaFilter**: Remove irrelevant tables (80% prob)
- **SchemaPermute**: Reorder elements
- **SQLTranslate**: Generate equivalent SQL forms

---

## 📊 Recommended Training Configuration (from Phase 1 Analysis)

### Hardware
- **GPU**: 8x A100 (80GB) or equivalent
- **Memory**: 640GB+ VRAM for parallel training
- **CPU**: 128+ cores recommended

### Hyperparameters for Mistral-Small-3.2-24B
```
Learning Rate: 1e-4
Epochs: 3-5
Batch Size: 1 per GPU
Gradient Accumulation: 4-8 steps
Max Sequence Length: 10,240 tokens
BF16 Precision: Yes
Flash Attention: Yes
Gradient Checkpointing: Yes
DeepSpeed: ZeRO-2 or ZeRO-3

LoRA Config:
- R: 512
- Alpha: 512
- Dropout: 0.05
- Target Modules: [q_proj, k_proj, v_proj, o_proj, up_proj, gate_proj, down_proj]
```

### Dataset Composition (Recommended)
- **Total Samples**: 5,000-10,000 (minimum), 20,000-50,000 (excellent)
- **Mix**:
  - 70% Standard NL2SQL samples
  - 20% Self-refinement (error correction) samples
  - 10% Candidate selection samples

---

## 🗂️ Data Pipeline (Step-by-Step)

### Step 1: Prepare Raw Data
```json
[
  {
    "db_id": "your_database",
    "question": "Natural language question",
    "evidence": "Optional reasoning",
    "SQL": "SELECT ... FROM ..."
  }
]
```
**Location**: `data/data_warehouse/your_dataset/raw_data/`

### Step 2: Create DB Connection Config
```json
{
  "host": "localhost",
  "port": 5432,
  "user": "user",
  "password": "pass",
  "dialect": "postgresql",
  "databases": {"your_database": "your_database"}
}
```
**Location**: `data/data_warehouse/your_dataset/db_conn.json`

### Step 3: Process Raw Data (adds M-Schema)
```bash
cd data
python data_processing.py \
  --raw_data_path data_warehouse/your_dataset/raw_data/raw_data.json \
  --db_conn_config data_warehouse/your_dataset/db_conn.json \
  --processed_data_dir data_warehouse/your_dataset/processed_data/
```

### Step 4: Assemble Training Data
Update `data/configs/datasets_all.json`:
```json
{
  "your_dataset": {
    "data_path": "data_warehouse/your_dataset/processed_data/raw_data_nl2postgresql.json",
    "sum_num": 5000,
    "task_name": "nl2postgresql",
    "data_aug": true
  }
}
```

Then run:
```bash
python data_assembler.py \
  --dataset_config_path configs/datasets_all.json \
  --save_path ../train/datasets/training_data_final.json
```

### Step 5: Train
```bash
cd train
bash askguru_sft.sh
```

### Step 6: Inference
```bash
cd evaluation
python sql_infer.py \
  --model_name_or_path /path/to/finetuned/model \
  --test_set_path test_data.json \
  --expr_version my_v1
```

### Step 7: Evaluate
```bash
bash sql_eval.sh \
  --pred_sql_path output/results.json \
  --test_sql_path test_data.json \
  --db_conn_config db_conn.json
```

---

## 🚀 Implementation Stages (Phased Approach)

### **Phase 1: Foundation (Week 1-2)**
- ✅ Framework migration (DONE)
- [ ] Understand existing codebase structure
- [ ] Set up development environment
- [ ] Run existing example with 221 samples
- [ ] Verify training/inference pipeline works

**Commands**:
```bash
cd askGuru-SQL
pip install -r requirements.txt
python data/data_assembler.py --dataset_config_path data/configs/datasets_example.json
cd train
python sft4askguru.py --help  # See all options
```

### **Phase 2: Data Collection (Week 3-6)**
- [ ] Collect 1,000-5,000 domain-specific NL-to-SQL samples
- [ ] Validate SQL queries on actual database
- [ ] Generate M-Schema from your database
- [ ] Create error-correction samples (20% of dataset)
- [ ] Set up data augmentation

**Minimum for training**: 1,000 samples  
**Recommended**: 5,000-10,000 samples

### **Phase 3: Training (Week 7-10)**
- [ ] Fine-tune on domain data with LoRA
- [ ] Monitor training metrics (loss, learning curve)
- [ ] Run evaluation against test set
- [ ] Collect failure cases for Phase 4

### **Phase 4: Refinement (Week 11-12)**
- [ ] Implement critic loop with error handling
- [ ] Add self-refinement training samples
- [ ] Test multi-candidate generation + selection
- [ ] Deploy to production

---

## 📋 Multi-Candidate Generation Workflow

### **Use Case**: Uncertain Outputs
When model generates SQL, use **ensemble approach**:

```python
# 1. Generate 5 candidates (beam search or sampling)
candidates = generator.generate_candidates_hybrid(question)

# 2. Execute each candidate
results = [execute(sql) for sql in candidates]

# 3. Use critic to select best
best_idx, best_sql = critic.select_best_candidate(
    question, schema, candidates, results
)

# 4. Return best SQL
return best_sql
```

### **Implementation**
See `evaluation/sql_multi_candidate_infer.py` for complete implementation.

**Run it**:
```bash
cd evaluation
python sql_multi_candidate_infer.py \
  --generator_path /path/to/model \
  --test_data_path test.json \
  --num_candidates 5 \
  --generation_method hybrid \
  --use_critic true
```

---

## 🔄 Self-Refinement (Critic Loop)

### **Architecture**
```
Question
  ↓
[Generate SQL]
  ↓
[Execute on DB]
  ├─→ Success → Return SQL
  └─→ Error → Capture error message
       ↓
    [Refine using error]
       ↓
    [Execute again]
       ├─→ Success → Return SQL
       └─→ Error → Attempt #3 (use candidate selection)
```

### **Training Data Format**
```json
{
  "db_name": "my_db",
  "question": "Find users from 2024",
  "pred_sql_res": [
    "SELECT * FROM user WHERE year = 2024",  // Wrong SQL
    "Column 'year' not found"  // Error message
  ],
  "sql": "SELECT * FROM users WHERE YEAR(created_at) = 2024",
  "db_schema": "...",
  "evidence": "..."
}
```

### **Task Type in Training**
```python
"task_name": "self_refine"  # Enables error-correction training
```

---

## 🎯 Oracle EBS Integration (Optional)

If you need Oracle EBS support:

1. **Use `oracle_ebs_implementation.py`** - Ready to use
2. **Three approaches**:
   - **Direct**: Train on Oracle EBS data (10K samples, slower)
   - **Conversion**: Train on PostgreSQL, auto-convert to Oracle (faster)
   - **Hybrid** ⭐: Train on PostgreSQL + fine-tune with 2K Oracle samples (best)

**Quick Start**:
```python
from oracle_ebs_implementation import OracleEBSDatasetConverter

converter = OracleEBSDatasetConverter()
converter.convert_dataset(
    'postgresql_data.json',
    'oracle_ebs_data.json'
)
```

---

## 🔗 Key Files Reference

| File | Purpose | Key Function |
|------|---------|--------------|
| `train/sft4askguru.py` | Training entry point | `train()` |
| `train/trainer/trainer.py` | Custom trainer | `DeepCustomTrainer` class |
| `data/data_processing.py` | Raw → Processed | `process_data()` |
| `data/data_assembler.py` | Processed → Training format | `assemble_dataset()` |
| `data/data_utils/m_schema.py` | Schema generation | `MSchema` class |
| `data/data_utils/prompt_utils.py` | Prompts for all tasks | `NL2SQLITE_TEMPLATE`, etc. |
| `evaluation/sql_infer.py` | Single-candidate inference | `inference_batch()` |
| `evaluation/sql_multi_candidate_infer.py` | Multi-candidate + critic | `MultiCandidateInference` |
| `oracle_ebs_implementation.py` | Oracle EBS support | `OracleEBSSQLConverter` |

---

## 📊 Performance Benchmarks (Reference)

### askGuru-SQL SOTA Results
- **BIRD-CRITIC-Open**: 44.37% (multi-dialect)
- **BIRD-CRITIC-PG**: 44.53% (PostgreSQL)
- **BIRD-CRITIC-Flash**: 48.5% (SQLite)

### Expected Results with Mistral-24B
- **With 5K samples**: 35-40% accuracy
- **With 10K samples**: 42-48% accuracy
- **With 50K samples**: 52-60% accuracy

*(Results depend on data quality, diversity, and domain relevance)*

---

## 🛠️ Development Workflow for askGuru-SQL

### 1. **Setup & Verification**
```bash
cd c:\Users\ALIENWARE\Projects\askGuru-SQL
pip install -r requirements.txt

# Verify with existing example data
cd data
python data_assembler.py --dataset_config_path configs/datasets_example.json
```

### 2. **Customize for Your Domain**
- [ ] Update `data/data_utils/prompt_utils.py` with your prompts
- [ ] Add domain-specific templates
- [ ] Modify `train/trainer/argument.py` for your hyperparameters

### 3. **Prepare Your Data**
- [ ] Collect NL-to-SQL pairs
- [ ] Create `data/data_warehouse/your_domain/`
- [ ] Add `raw_data.json` and `db_conn.json`
- [ ] Run `data_processing.py` and `data_assembler.py`

### 4. **Train Your Model**
```bash
cd train
# Edit askguru_sft.sh with your config
bash askguru_sft.sh
```

### 5. **Deploy & Evaluate**
```bash
cd evaluation
python sql_infer.py --model_name_or_path ./output/checkpoint-final
python sql_eval.sh --pred_sql_path output/results.json
```

---

## 💡 Suggestions & Next Steps

### **Recommended Approach for askGuru-SQL**

1. **Start with Phase 1 (Weeks 1-2)**
   - Don't train yet. Just understand the framework.
   - Run existing code with 221 samples
   - Read `prompt_utils.py` to understand prompt engineering
   - Study `m_schema.py` to understand schema representation

2. **Decide on Domain (Week 3)**
   - What database systems? (PostgreSQL, MySQL, Oracle?)
   - What industry? (Finance, HR, E-commerce?)
   - Estimate data collection effort

3. **Collect Minimal Dataset (Week 4-5)**
   - Start with just 500-1000 samples
   - High quality > High quantity
   - Include diverse SQL operations (JOIN, GROUP BY, HAVING, etc.)
   - Create 20% error-correction samples

4. **Quick Training & Test (Week 6)**
   - Train on 500 samples to verify pipeline
   - Check if loss decreases, metrics improve
   - Identify problems early

5. **Scale Up (Week 7+)**
   - Expand to 5K-10K samples
   - Implement critic loop
   - Deploy multi-candidate generation
   - Add self-refinement training

---

## ⚠️ Common Pitfalls & Solutions

| Problem | Cause | Solution |
|---------|-------|----------|
| Training crashes on CUDA OOM | Batch size too large | Reduce batch_size or use DeepSpeed ZeRO-3 |
| Model generates invalid SQL | Poor training data | Validate all SQL on actual DB before training |
| Low accuracy on test set | Dataset too small | Collect more samples (5K minimum) |
| Slow training | No distributed training | Use `DeepSpeed` with multiple GPUs |
| Inference is slow | Loading full model each time | Use LoRA adapter + merge once |
| Critic selection fails | Critic not trained | Train critic on selection task first |

---

## 📚 Documentation Includes

- ✅ `BIRD_CRITIC_GUIDE.md` - Evaluation benchmarks & self-refinement
- ✅ `DATASET_PREPARATION_GUIDE.md` - Data pipeline & M-Schema
- ✅ `COMPLETE_END_TO_END_GUIDE.md` - Production workflow
- ✅ `ORACLE_EBS_INTEGRATION_GUIDE.md` - Oracle EBS support (optional)
- ✅ `oracle_ebs_implementation.py` - Ready-to-use Oracle converter

---

## 🚀 Quick Start Command

```bash
cd c:\Users\ALIENWARE\Projects\askGuru-SQL

# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare data (using example)
cd data
python data_assembler.py --dataset_config_path configs/datasets_example.json

# 3. Check training data
ls -la ../train/datasets/

# 4. Look at framework structure
cd ..
find . -name "*.py" | head -20
```

---

## 📞 Key Contacts & References

- **Original Framework**: askGuru-SQL (Apache 2.0)
- **Model**: Mistral-Small-3.2-24B-Instruct-2506
- **Technologies**: 
  - Transformers (Hugging Face)
  - LoRA (PEFT)
  - DeepSpeed
  - SQLGlot (for dialect conversion)
  - PySQL (for execution)

---

**You're now ready to build askGuru-SQL with a production-grade NL-to-SQL framework!**
