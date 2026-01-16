# Oracle EBS NL2SQL Data Pipeline Plan

**Date:** 2025-12-27  
**Purpose:** Define data consolidation and processing strategy for SFT training  
**Status:** PLANNING (awaiting approval before implementation)

---

## Executive Summary

The `data/data_warehouse/oracle_train/` folder contains **multiple complementary datasets** created for different purposes:

1. **Main Training Dataset** (1.82 MB) - `training_data_sets_4jul2025_v5_askguru_compatible.jsonl`
2. **Schema Learning Dataset** (114 KB) - `cleaned_main_tables_15_06_2025.askguru.train.jsonl`
3. **Quality Corrections** (8.81 KB) - `corrections.jsonl`
4. **Reference Data:** joins, keywords, table schemas, gold standard SQLs, schema definitions

**Goal:** Consolidate these into a single production-ready dataset in **askGuru conversation format** with rich **RAG context**.

---

## Current Dataset Inventory

### 1. Training Data Files

| File | Size | Format | Status | Content |
|------|------|--------|--------|---------|
| `training_data_sets_4jul2025_v5_askguru_compatible.jsonl` | 1.82 MB | `{instruction, input, output}` | ⚠️ NOT askGuru format | Generic instruction + Q&A pairs (~2600 lines) |
| `cleaned_main_tables_15_06_2025.askguru.train.jsonl` | 114 KB | `{instruction, input, output}` | ✅ Schema format | Schema learning examples (28 lines) |
| `corrections.jsonl` | 8.81 KB | `{id, question, candidate_sql, corrected_sql}` | ⚠️ Correction format | High-quality corrections (3 lines) |

### 2. Reference/Support Files

| File | Purpose | Usage |
|------|---------|-------|
| `joins.json` | 82 join paths | Build RAG context (foreign key relationships) |
| `joins_patch.json` | Additional joins | Enrich relationship mapping |
| `keywords.json` | SQL keywords/mappings | Business term → DB mapping |
| `*.json` (table schemas) | 27 table definitions | Table structure, columns, FK relationships |
| `gold_standard_sqls.md` | 2000+ reference SQLs | Validation & quality benchmarks |
| `all_27_table_selects.sql` | Basic SELECT examples | Simple query references |
| `askGuru_m-schema_converted.txt` | Full schema documentation | Comprehensive schema reference |

---

## Data Format Analysis

### Current Format 1: Generic Instruction-Input-Output
**File:** `training_data_sets_4jul2025_v5_askguru_compatible.jsonl`

```json
{
  "instruction": "Generate SQL using PostgreSQL schema",
  "input": "List all ledgers with their name and currency code, PERIOD_TYPE",
  "output": "SELECT name AS ledger_name, currency_code,PERIOD_SET_NAME,ACCOUNTED_PERIOD_TYPE,LEDGER_CATEGORY_CODE FROM gl_ledgers"
}
```

**Issues:**
- ❌ Not in askGuru `conversations` format
- ❌ Generic instruction (doesn't mention Oracle EBS)
- ❌ NO RAG context (no schema hints, no join information)
- ⚠️ Some duplicate examples (rows 1-5 are identical)
- ⚠️ SQL quality varies (some formatting issues, some missing spaces)

**Conversion Required:** `{instruction, input, output}` → `{id, sql_type, conversations, meta}`

### Current Format 2: Schema Learning
**File:** `cleaned_main_tables_15_06_2025.askguru.train.jsonl`

```json
{
  "instruction": "### Instruction: Learn and memorize schema...[full m-schema definition]...",
  "input": "",
  "output": "OK"
}
```

**Issues:**
- ❌ Not question-answering examples
- ✅ Rich schema definitions (m-schema format)
- ❌ Input/output are empty/OK (no actual training signal)

**Usage:** Extract schema information for RAG context, don't use as training pairs

### Current Format 3: Corrections
**File:** `corrections.jsonl`

```json
{
  "id": 1759227258067,
  "question": "Give me closing balances for Jan and June 2003...",
  "candidate_sql": "[potentially incorrect SQL]",
  "corrected_sql": "[corrected, better SQL]"
}
```

**Issues:**
- ⚠️ Only 3 examples (very small)
- ✅ High quality corrections
- ❌ Format includes "candidate" SQL which we don't want in training

**Usage:** Use only `{question, corrected_sql}` pairs

---

## Data Quality Issues Identified

### Issue 1: Duplicates in Main Training File
```
Lines 1-5: Identical examples (5 copies of same Q&A)
```
**Action:** Deduplication required

### Issue 2: Missing RAG Context
**All files:** No context about:
- Which tables are needed
- What joins are available
- What columns are relevant
- Business mapping (aliases, synonyms)

**Action:** Enrich with RAG context from:
- `joins.json` (foreign key relationships)
- Table schema files (column definitions)
- `keywords.json` (business mappings)

### Issue 3: SQL Quality Variance
Some examples have formatting issues:
- Missing spaces after commas
- Inconsistent capitalization
- Oracle vs PostgreSQL dialect mix

**Action:** Normalize SQL output

### Issue 4: Mixed Database Dialects
**Problem:** `training_data_sets_4jul2025_v5_askguru_compatible.jsonl` says:
- Instruction: "PostgreSQL schema"
- But output: Oracle SQL (e.g., `NVL()`, `TO_CHAR()`)

**Action:** Normalize instructions to mention Oracle EBS

---

## Proposed Data Pipeline

### Phase 1: Data Analysis & Inventory (CURRENT)

✅ Understand file formats and content
✅ Identify issues and overlaps
✅ Create consolidation plan

### Phase 2: Data Extraction & Transformation

#### Step 2.1: Extract Main Training Examples
```
Source: training_data_sets_4jul2025_v5_askguru_compatible.jsonl

Process:
1. Load all ~2600 lines
2. Deduplicate (remove exact duplicates)
3. Map: {instruction, input, output} → {question, sql_gold}
4. Normalize SQL (formatting, Oracle dialect verification)
5. Label: dataset_id = "oracle_ebs_v1_main"
```

**Output:** ~2400-2500 deduplicated (question, sql_gold) pairs

#### Step 2.2: Extract Corrections
```
Source: corrections.jsonl

Process:
1. Load all 3 lines
2. Map: {question, corrected_sql} → {question, sql_gold}
3. Label: dataset_id = "oracle_ebs_v1_corrections"
4. Priority: HIGH (manual corrections = high quality)
```

**Output:** 3 high-quality (question, sql_gold) pairs

#### Step 2.3: Build RAG Context Database
```
Source: joins.json, keywords.json, *.json (table schemas)

Process:
1. Parse table schemas (*.json files):
   - Extract table names, columns, types, FK relationships
   - Build column → table mapping
2. Parse joins.json:
   - Extract all valid join paths
   - Build join graph
3. Parse keywords.json:
   - Extract business mappings
4. Build RAG generator function:
   For each (question, sql_gold) pair:
   - Parse SQL to identify tables involved
   - Look up join paths between tables
   - Generate context sections:
     * Candidate Tables
     * Join Graph
     * Relevant Columns
     * Business Synonyms
```

**Output:** RAG context enrichment function

#### Step 2.4: Enrich with RAG Context
```
Process:
1. For each (question, sql_gold) pair:
   - Parse the SQL to extract tables used
   - Generate RAG context (joins, columns, relationships)
   - Create rag_context field
2. Quality check:
   - Verify all tables in SQL appear in schemas
   - Verify all joins are valid
   - Flag problematic examples
```

**Output:** (question, sql_gold, rag_context) tuples

### Phase 3: Format Conversion to askGuru

#### Step 3.1: Required Prompt Template Structure

The user prompt **MUST** follow this exact structure (from REQUIREMENTS.md section 7 + requriment_document.md section 6.3):

```
1. INSTRUCTION HEADER
   You are a Text-to-SQL generator for Oracle EBS.
   Return ONLY Oracle SQL. No markdown. No explanations. No comments.

2. RAG CONTEXT (structured sections in order):
   # Candidate Tables
   - TABLE_NAME: Brief description (columns: col1, col2, ...)
   - ...
   
   # Join Graph
   - TABLE_A.col1 = TABLE_B.col2
   - TABLE_B.col3 = TABLE_C.col4
   - ...
   
   # Relevant Columns
   - TABLE_NAME.column_name: Type (e.g., VARCHAR2, NUMBER)
   - ...
   
   # Synonyms/Business Mapping
   - "business_term" → TABLE.column
   - ...
   
   # Security Filters (if applicable)
   - Always filter by: WHERE org_id = ? AND set_of_books_id = ?
   - ...

3. QUESTION
   [User Question]
   [The actual question from example]
```

**Critical Requirements:**
- ✅ No markdown fences (```) in prompt or SQL
- ✅ Only SELECT statements (no DML/DDL)
- ✅ All tables in candidate list must be from joins.json or table schemas
- ✅ All joins must be documented and valid
- ✅ Column references must exist in table schemas
- ⚠️ Keep assistant content as **SQL ONLY** - no explanations, no markdown

#### Step 3.2: Convert to Conversations Format

```python
def create_conversation(question: str, sql_gold: str, rag_context: str) -> List[Dict]:
    """
    Create askGuru conversations format.
    
    Input:
      - question: Natural language question
      - sql_gold: Ground truth Oracle SQL
      - rag_context: Generated RAG context string with sections
      
    Output:
      [
        {"role": "user", "content": "<full prompt with rules + rag_context + question>"},
        {"role": "assistant", "content": "<sql_gold ONLY - no markdown>"}
      ]
    """
    user_prompt = f"""You are a Text-to-SQL generator for Oracle EBS.
Return ONLY Oracle SQL. No markdown. No explanations.

{rag_context}

[User Question]
{question}"""
    
    return [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": sql_gold.strip()}
    ]
```

**Assistant Content Requirements:**
- ✅ ONLY SQL - no markdown fences
- ✅ ONLY SELECT statements
- ✅ No comments or explanations
- ❌ REJECT if contains: "```", "SELECT\n--", DML/DDL keywords
- Training script validates with `_basic_sql_only_check()` (blocks "```" and non-SELECT)

#### Step 3.3: Create Metadata

```json
{
  "id": "oracle_ebs_v1_000001",
  "sql_type": "oracle",
  "conversations": [
    {
      "role": "user",
      "content": "You are a Text-to-SQL generator for Oracle EBS...\n# Candidate Tables\n- GL_LEDGERS: ...\n\n# Join Graph\n...\n[User Question]\n[question]"
    },
    {
      "role": "assistant",
      "content": "SELECT ... FROM gl_ledgers WHERE ..."
    }
  ],
  "meta": {
    "dataset_id": "oracle_ebs_v1",
    "source_file": "training_data_sets_4jul2025_v5_askguru_compatible.jsonl",
    "has_joins": true,
    "num_tables": 5,
    "quality_tier": "primary",
    "verified": true,
    "example_tables": ["GL_LEDGERS", "GL_PERIODS"],
    "rag_context_length": 1850,
    "prompt_token_estimate": 320,
    "tags": ["ledger", "accounting", "financial"]
  }
}
```

### Phase 4: Training Data Preparation & Label Masking

#### Step 4.1: Training Label Alignment

The training script (`sft_oracle_llama70b_lora.py`) uses **label masking** (from training code lines 95-115):

```python
# Key masking logic in training:
text = tokenizer.apply_chat_template(conversations, tokenize=False, add_generation_prompt=False)
enc = tokenizer(text, truncation=True, max_length=model_max_length)

target = conversations[1]["content"]  # Assistant SQL content
idx = text.find(target)
target_idx = enc.char_to_token(idx) if idx >= 0 else None

# Mask all tokens BEFORE assistant content
labels = enc["input_ids"].copy()
if target_idx is not None:
    labels[:target_idx] = [IGNORE_TOKEN_ID] * target_idx  # Mask prompt
else:
    # Fallback: train on all tokens if alignment fails
    pass
```

**CRITICAL**: This means:
- ✅ Training ONLY optimizes on the assistant's SQL tokens
- ✅ Prompt tokens are masked with `IGNORE_TOKEN_ID` (-100)
- ✅ Loss is computed ONLY from assistant content
- ❌ If target SQL span cannot be found, example may train on full sequence (WARNING)

**For our pipeline:**
- Ensure `create_conversation()` produces exact target text (SQL only)
- Ensure SQL is easily locatable in the full prompt text
- Avoid whitespace inconsistencies that might break alignment

#### Step 4.2: Quality Assurance

#### Step 4.2.1: Validation Checks
```
For each example:
1. ✅ Conversations format valid (has user + assistant roles)
2. ✅ Assistant content is PURE SQL (no markdown, no explanations)
3. ✅ Assistant content starts with SELECT or WITH (no DML/DDL)
4. ✅ RAG context non-empty and properly formatted
5. ✅ All tables in Candidate Tables section exist in schemas
6. ✅ All joins in Join Graph are in joins.json
7. ✅ All columns in Relevant Columns section exist in schemas
8. ✅ Business mappings in Synonyms section are valid
9. ✅ Total user prompt length is reasonable (6.5k-16k chars recommended)
10. ❌ Flag/reject if contains markdown (```), comments (--), or invalid spans
```

**Rejection Criteria:**
- SQL contains "```" (markdown)
- SQL is empty or not SELECT/WITH
- SQL references undefined tables/columns
- Target SQL span cannot be aligned in prompt
- User prompt exceeds 16,000 characters (token limit risk)

#### Step 4.2.2: Quality Metrics
```
Track:
- Total examples: X (target: ~2,400-2,500)
- Examples with joins: Y (% multi-table)
- Examples with full RAG context: Z
- Examples with quality warnings: W
- Examples rejected: R (and reasons)
- Average user prompt length (chars + tokens)
- Average SQL output length (tokens)
- Distribution by table/domain (AP, GL, PO, MTL, HR, RCV)
- High-quality tier (corrections): 3 examples
- Primary tier (main training): ~2,400 examples
```

#### Step 4.2.3: Validation Output
```json
{
  "validation_report": {
    "total_examples": 2503,
    "valid_examples": 2480,
    "rejected_examples": 23,
    "rejection_reasons": {
      "markdown_in_sql": 8,
      "undefined_tables": 7,
      "alignment_failed": 5,
      "invalid_syntax": 3
    },
    "quality_tiers": {
      "high": 3,        # corrections.jsonl
      "primary": 2397,  # main training data
      "secondary": 80   # schema learning converted
    },
    "avg_user_prompt_length": 1850,
    "avg_sql_length": 95,
    "tables_covered": 27,
    "join_examples": 1650,
    "single_table_examples": 830
  }
}
```

### Phase 5: Train/Val/Test Split

```
Total examples: ~2500

Splits:
- Training: 80% (~2000 examples)
- Validation: 10% (~250 examples)
- Test: 10% (~250 examples)

Strategy:
1. Stratify by:
   - Number of tables (single table vs multi-table)
   - Domain (AP, GL, PO, MTL, HR, RCV)
   - Has joins (yes/no)
2. Ensure corrections.jsonl distributed across all splits
3. Store as separate files
```

**Output:**
- `oracle_sft_conversations_train.json`
- `oracle_sft_conversations_val.json`
- `oracle_sft_conversations_test.json`

---

## Data Schema: Input & Output

### Input to Pipeline

```
data/data_warehouse/oracle_train/
├── training_data_sets_4jul2025_v5_askguru_compatible.jsonl    (2600 examples)
├── corrections.jsonl                                        (3 examples)
├── joins.json                                               (FK relationships)
├── keywords.json                                            (business mappings)
├── *.json (27 table schemas)                                (schema definitions)
└── gold_standard_sqls.md                                    (reference SQLs)
```

### Output from Pipeline

```
data/oracle_sft_conversations/
├── oracle_sft_conversations_full.json     (~2500 total, askGuru format)
├── oracle_sft_conversations_train.json    (~2000, 80%)
├── oracle_sft_conversations_val.json      (~250, 10%)
├── oracle_sft_conversations_test.json     (~250, 10%)
├── data_quality_report.json               (metrics + validation)
└── rag_context_samples.json               (example contexts)
```

### askGuru Conversation Format (Output)

```json
{
  "id": "oracle_ebs_v1_000001",
  "sql_type": "oracle",
  "conversations": [
    {
      "role": "user",
      "content": "You are a Text-to-SQL generator for Oracle EBS.\nReturn ONLY Oracle SQL. No markdown. No explanations.\n\n# Context (retrieved)\n## Candidate Tables\n- GL_LEDGERS: GL ledger master (ledger_id, name, currency_code, ...)\n- GL_PERIODS: Period definitions (period_name, period_type, ...)\n\n## Join Graph\n- GL_LEDGERS.PERIOD_SET_NAME = GL_PERIODS.PERIOD_SET_NAME\n- GL_LEDGERS.ACCOUNTED_PERIOD_TYPE = GL_PERIODS.PERIOD_TYPE\n\n## Relevant Columns\n- GL_LEDGERS: name, currency_code, PERIOD_SET_NAME, ACCOUNTED_PERIOD_TYPE\n- GL_PERIODS: PERIOD_NAME, PERIOD_SET_NAME, PERIOD_TYPE\n\n# Question\nList all ledgers with their name and currency code, PERIOD_TYPE"
    },
    {
      "role": "assistant",
      "content": "SELECT name AS ledger_name, currency_code, PERIOD_SET_NAME, ACCOUNTED_PERIOD_TYPE, LEDGER_CATEGORY_CODE FROM gl_ledgers"
    }
  ],
  "meta": {
    "dataset_id": "oracle_ebs_v1",
    "source_file": "training_data_sets_4jul2025_v5_askguru_compatible.jsonl",
    "tables": ["GL_LEDGERS", "GL_PERIODS"],
    "has_joins": true,
    "has_aggregation": false,
    "complexity": "simple",
    "quality_tier": "primary",
    "deduplication_id": null
  }
}
```

---

## Dataset Size Requirements (from REQUIREMENTS.md + DATASET_PREPARATION_GUIDE.md)

### For LLaMA-3.3-70B-Instruct

Based on REQUIREMENTS.md section 3 and DATASET_PREPARATION_GUIDE.md guidelines:

| Metric | Value | Status |
|--------|-------|--------|
| **Model Size** | 70B | Large |
| **Minimum Samples** | 2,000 | ✅ We have ~2,500 |
| **Recommended** | 10,000-20,000 | ⚠️ We have ~2,500 (borderline) |
| **Optimal** | 20,000-50,000+ | ❌ Not available |

**Current Status:**
- **Available**: ~2,500 deduplicated examples
- **Assessment**: Acceptable for **proof-of-concept & domain-specific finetuning** but below ideal for general performance
- **Recommendation**: Use all 2,500 examples + consider data augmentation if quality concerns arise

**Data Composition:**
- High-quality corrections: 3 examples (100% oracle quality)
- Main training: ~2,397 examples (primary tier)
- Secondary/schema learning: ~100 examples (converted or optional)

---

## Data Augmentation Strategy (APPROVED - Hybrid Approach)

### Rationale
- Current dataset: ~2,500 examples (borderline for 70B model)
- Recommendation: 10,000-20,000 for optimal performance
- **Solution:** Safe augmentation using only real data (no synthetic generation)

### Target Dataset Size
- **Original:** 2,500 examples
- **After augmentation:** 6,000-7,500 examples (2.4-3x multiplier)
- **Method:** Schema permutation + Synonym swapping
- **Quality:** 100% safe (uses only real tables, columns, mappings)

### Augmentation Techniques (IMPLEMENTED IN PHASE 2)

#### Technique 1: Schema Permutation (Multiplier: 2-2.5x)
```
For each training example:
- Identify candidate tables in RAG context
- Generate 1-2 variations with tables listed in different orders
- Generate variations with different column orderings
- Keep SQL identical (semantic equivalence guaranteed)

Example:
Original Candidate Tables:
- GL_LEDGERS
- GL_PERIODS

Augmented Variation 1:
- GL_PERIODS
- GL_LEDGERS

Result: Same SQL query, different table ordering in RAG context
Safety: ✅ HIGH (uses only real tables)
```

#### Technique 2: Synonym Swapping (Multiplier: 1.5-2x)
```
For each training example:
- Identify business terms in question (from keywords.json)
- Generate 1-2 variations with synonyms
- Keep SQL unchanged (still generates same output)

Example:
Original Question:
"List all ledgers with their period type"

Augmented Variation 1:
"Show me the ledger names and accounting period type"

Augmented Variation 2:
"Find all ledger records with their period type"

Result: Different phrasing, same SQL output
Safety: ✅ HIGH (uses only real synonyms from keywords.json)
```

#### Technique 3: SQL Normalization (Multiplier: 1.1-1.2x) - OPTIONAL
```
For each training example:
- Generate 1 variation with table aliases
- Generate 1 variation with IN instead of = for single values
- Keep semantic meaning identical

Example:
Original SQL:
SELECT name FROM gl_ledgers WHERE currency_code = 'USD'

Augmented Variation 1:
SELECT gl.name FROM gl_ledgers gl WHERE currency_code = 'USD'

Augmented Variation 2:
SELECT name FROM gl_ledgers WHERE currency_code IN ('USD')

Result: Different SQL syntax, same semantic meaning
Safety: ⚠️ MEDIUM (validate against Oracle syntax rules)
```

### Implementation Plan

#### Phase 2 Enhancement: Add Augmentation Functions
```python
def augment_schema_permutation(
    example: Dict[str, Any],
    num_variations: int = 2
) -> List[Dict[str, Any]]:
    """
    Generate schema permutation variations
    
    Input: One training example with RAG context
    Output: Original + 1-2 permutations with different table orderings
    """

def augment_synonym_swap(
    example: Dict[str, Any],
    keywords: Dict[str, str],
    num_variations: int = 2
) -> List[Dict[str, Any]]:
    """
    Generate synonym swap variations
    
    Input: One training example with question
    Output: Original + 1-2 variations with synonyms swapped
    """

def augment_sql_normalization(
    example: Dict[str, Any],
    num_variations: int = 1
) -> List[Dict[str, Any]]:
    """
    Generate SQL normalization variations (optional)
    
    Input: One training example
    Output: Original + 1 variation with different SQL syntax
    """

def apply_augmentation(
    examples: List[Dict[str, Any]],
    config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Apply augmentation to entire dataset
    
    Config:
    {
        "enable_schema_permutation": True,
        "schema_permutation_variations": 2,
        "enable_synonym_swap": True,
        "synonym_swap_variations": 2,
        "enable_sql_normalization": False,
        "sql_normalization_variations": 1
    }
    
    Returns: Augmented dataset with new examples
    """
```

#### Phase 2.5 (New): Augmentation & Deduplication
```
Step 2.5.1: Apply augmentation to main training data
- Schema permutation: 2,397 × 2 = 4,794 variations
- Synonym swapping: 2,397 × 2 = 4,794 variations
- Combined (with overlap removal): ~2,397 × 2.4 = 5,753 examples

Step 2.5.2: Keep corrections.jsonl unchanged
- 3 high-quality examples (no augmentation needed)

Step 2.5.3: Deduplicate augmented dataset
- Remove any exact duplicates introduced by augmentation
- Cross-check with original corrections

Step 2.5.4: Quality validation
- Verify all augmented examples have valid RAG context
- Verify SQL alignment works for all variations
- Track augmentation source (original + technique)
```

### Expected Results

**Dataset Size Progression:**
```
Raw data:              ~2,600 examples
After deduplication:   ~2,400 examples
After augmentation:    ~6,000-7,500 examples (2.4-3x)
Final split:
  - Train (80%):       4,800-6,000 examples
  - Val (10%):         600-750 examples
  - Test (10%):        600-750 examples
```

**Quality Guarantees:**
- ✅ All augmentations use real data (tables, columns, mappings)
- ✅ SQL output unchanged (same ground truth)
- ✅ RAG context valid (uses joins.json + schemas)
- ✅ Keywords validated against keywords.json
- ✅ Augmentation source tracked in metadata

### Validation Strategy

#### Step 1: Augmentation Quality Check
```
For each augmented example:
1. ✅ Original and augmented have same SQL output
2. ✅ RAG context uses only real tables (from schemas)
3. ✅ Joins are valid (from joins.json)
4. ✅ Keywords exist (from keywords.json)
5. ✅ No exact duplicates with originals
```

#### Step 2: Training Compatibility
```
For each augmented example:
1. ✅ Can create conversation format (has user + assistant)
2. ✅ Target SQL aligns in prompt (label masking works)
3. ✅ Prompt length reasonable (6.5k-16k chars)
4. ✅ No markdown or invalid syntax
```

#### Step 3: Metrics Tracking
```
Report:
- Total original examples: 2,400
- Total augmented examples: 2,600-3,100
- Augmentation multiplier achieved: 2.4-3x
- Deduplication rate: X%
- Failed augmentations: Y (reasons tracked)
- Final dataset size: 6,000-7,500
```

### Configuration (Added to oracle_sft_config.yaml)

```yaml
augmentation:
  enabled: true
  techniques:
    schema_permutation:
      enabled: true
      variations_per_example: 2
      multiplier_expected: 2.0-2.5
    synonym_swap:
      enabled: true
      variations_per_example: 2
      multiplier_expected: 1.5-2.0
    sql_normalization:
      enabled: false
      variations_per_example: 1
      multiplier_expected: 1.1-1.2
  
  # Combined effect
  total_multiplier_target: 2.4-3.0
  target_final_size: 6000-7500
  deduplication_enabled: true
  
  validation:
    check_sql_equivalence: true
    check_rag_validity: true
    check_keyword_mapping: true
    check_alignment: true
```

---

## Guide Reference: What Each Document Requires

### 1. REQUIREMENTS.md (oracle_askguru_llama_requirements_and_code)
**Applies to:** Entire project workflow (most specific)

**Key Requirements:**
- ✅ askGuru conversations format with user/assistant roles
- ✅ Assistant content: Oracle SQL only
- ✅ User content: Rules + RAG context + Question
- ✅ LoRA training on BF16 base
- ✅ 4-bit quantization (AWQ preferred, GPTQ fallback)
- ✅ vLLM serving with 4 concurrent requests
- ✅ SQL guardrail + one retry on error
- **Dataset format:** Section 7 (prompt format with structured sections)

### 2. requriment_document.md
**Applies to:** Training & packaging architecture (more detailed)

**Additional Requirements:**
- ✅ Deepspeed ZeRO-3 for training
- ✅ Default `model_max_length=8192`
- ✅ LoRA target modules: q_proj,k_proj,v_proj,o_proj,up_proj,gate_proj,down_proj
- ✅ Manifest tracking (base model, adapter, dataset version, quantization config)
- **Dataset:** Sections 6.1-6.3 (detailed dataset builder requirements)

### 3. DATASET_PREPARATION_GUIDE.md
**Applies to:** Data format & processing pipeline

**Key Data Structures:**
- ❌ M-Schema format: Optional (we use simpler RAG context instead)
- ✅ Training format: `{id, conversations, sql_type}`
- ✅ Conversations: `[{role, content}, ...]`
- **Dataset size:** Guidelines for model sizes (Section 📦)

### 4. ORACLE_EBS_INTEGRATION_GUIDE.md
**Applies to:** Oracle-specific SQL patterns

**Key Requirements:**
- ✅ Use Oracle EBS-specific patterns (module prefixes: AR_, AP_, GL_, PO_, etc.)
- ✅ Include security filters (org_id, set_of_books_id)
- ✅ Recognize Oracle functions (NVL, TRUNC, SYSDATE, etc.)
- ✅ Handle 1:M relationships (Oracle EBS specific)

### 5. COMPLETE_END_TO_END_GUIDE.md
**Applies to:** Inference & critic loop (post-training)

**Note:** Not critical for data pipeline but useful for later phases (evaluation, critic loop)

---

---

## Critical Decision Points (Awaiting User Approval)

### Decision 1: Schema Learning Examples (28 examples)
From `cleaned_main_tables_15_06_2025.askguru.train.jsonl`:
- **Option A:** Extract schema definitions, use ONLY for RAG context enrichment (not training)
- **Option B:** Convert to training examples (may be lower quality for training)
- **Option C:** Skip entirely, use only main training + corrections

**Recommendation:** Option A (use for context enrichment, not training)

### Decision 2: Quality Tier Weighting
The 3 corrections.jsonl examples are manually verified:
- **Option A:** Weight them equally (1x like main training)
- **Option B:** Weight higher during training (2x-5x) to emphasize quality
- **Option C:** Use only for validation/test set (not training)

**Recommendation:** Option B (weight 2-3x higher during training)

### Decision 3: RAG Context Scope
For each SQL, which related tables to include:
- **Option A:** ONLY tables used in the SQL query
- **Option B:** Used tables + directly joined tables (full join path)
- **Option C:** All related tables from the domain module

**Recommendation:** Option B (used + directly joined tables)

### Decision 4: Security Filters
Oracle EBS requires org_id and set_of_books_id in WHERE clauses:
- **Option A:** Always include security filter hints in RAG context
- **Option B:** Include only when relevant to the question domain
- **Option C:** Skip security filters (not reflected in training data)

**Recommendation:** Option B (domain-aware inclusion)

### Decision 5: Data Augmentation
Given ~2,500 examples (borderline for 70B model):
- **Option A:** Use all examples as-is, accept quality variance
- **Option B:** Implement simple augmentation (schema permutation, synonym swapping)
- **Option C:** Limit to high-quality examples only (~500), risk under-training

**Recommendation:** Option A initially, move to Option B if quality concerns arise

---

## Implementation Steps

### Step 1: Create Data Pipeline Module

**File:** `custom_oracle_llama/build_oracle_sft_dataset.py`

**Required Functions:**

```python
# ============ LOAD RAW DATA ============
def load_main_training_data(filepath: str) -> List[Dict[str, str]]:
    """
    Load training_data_sets_4jul2025_v5_askguru_compatible.jsonl
    
    Returns: [{instruction, input, output}, ...]
    Maps to: {question, sql_gold}
    """

def load_corrections(filepath: str) -> List[Dict[str, str]]:
    """
    Load corrections.jsonl
    
    Returns: [{id, question, candidate_sql, corrected_sql}, ...]
    Uses: {question, corrected_sql} as {question, sql_gold}
    """

def load_joins(filepath: str) -> List[str]:
    """
    Load joins.json
    
    Returns: ["TABLE_A.col = TABLE_B.col", ...]
    """

def load_table_schemas(folder: str) -> Dict[str, Dict]:
    """
    Load all *.json table schema files
    
    Returns: {
        "GL_LEDGERS": {
            "columns": {
                "ledger_id": {"type": "NUMBER", "pk": True},
                "name": {"type": "VARCHAR2", "pk": False},
                ...
            },
            "description": "GL ledger master table"
        },
        ...
    }
    """

def load_keywords(filepath: str) -> Dict[str, str]:
    """
    Load keywords.json
    
    Returns: {"business_term": "TABLE.column", ...}
    """

# ============ TRANSFORM DATA ============
def deduplicate_examples(examples: List[Dict]) -> List[Dict]:
    """
    Remove exact duplicate {question, sql_gold} pairs
    
    Returns: Deduplicated list, keeps first occurrence
    """

def normalize_sql(sql: str) -> str:
    """
    Normalize SQL formatting:
    - Consistent whitespace
    - Uppercase keywords (SELECT, FROM, WHERE, etc.)
    - Remove trailing semicolons
    - Verify Oracle dialect (NVL, SYSDATE, etc. OK)
    """

def extract_tables_from_sql(sql: str, schemas: Dict[str, Dict]) -> List[str]:
    """
    Parse SQL to identify tables used
    
    Returns: ["GL_LEDGERS", "GL_PERIODS", ...]
    Validates against schemas dict
    Raises: ValueError if undefined tables found
    """

def find_join_paths(tables: List[str], joins: List[str]) -> List[str]:
    """
    Find FK relationships between given tables
    
    Input: ["GL_LEDGERS", "GL_PERIODS"]
    Output: ["GL_LEDGERS.PERIOD_SET_NAME = GL_PERIODS.PERIOD_SET_NAME", ...]
    """

def build_rag_context(
    question: str,
    tables: List[str],
    joins: List[str],
    schemas: Dict[str, Dict],
    keywords: Dict[str, str]
) -> str:
    """
    Generate structured RAG context string
    
    Output format:
    # Candidate Tables
    - TABLE_NAME: description (columns: col1, col2)
    
    # Join Graph
    - TABLE_A.col = TABLE_B.col
    
    # Relevant Columns
    - TABLE.column: Type
    
    # Synonyms/Business Mapping
    - "term" → TABLE.column
    """

# ============ CREATE TRAINING FORMAT ============
def create_conversation(
    question: str,
    sql_gold: str,
    rag_context: str
) -> Dict[str, Any]:
    """
    Create single askGuru conversation entry
    
    Returns: {
        "role": "user/assistant",
        "content": "..."
    }
    """

def build_training_example(
    example_id: str,
    question: str,
    sql_gold: str,
    rag_context: str,
    metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create complete training example in askGuru format
    
    Returns: {
        "id": "oracle_ebs_v1_000001",
        "sql_type": "oracle",
        "conversations": [{user}, {assistant}],
        "meta": {...}
    }
    """

# ============ VALIDATE & QA ============
def validate_example(
    example: Dict[str, Any],
    schemas: Dict[str, Dict],
    joins: List[str]
) -> Dict[str, Any]:
    """
    Validate a training example
    
    Returns: {
        "valid": bool,
        "errors": [list of validation errors],
        "warnings": [list of warnings],
        "alignment_test": bool  # Can target SQL be found in prompt?
    }
    """

def compute_quality_metrics(examples: List[Dict]) -> Dict[str, Any]:
    """
    Compute dataset-level quality metrics
    
    Returns: {
        "total": int,
        "valid": int,
        "rejected": int,
        "avg_prompt_length": float,
        "avg_sql_length": float,
        "join_examples": int,
        "tables_covered": set,
        ...
    }
    """

# ============ MAIN PIPELINE ============
def build_dataset(
    config: Dict[str, str]  # Config with input/output paths
) -> Dict[str, Any]:
    """
    End-to-end data pipeline orchestration
    
    Returns: {
        "train_data": [{examples...}],
        "val_data": [{examples...}],
        "test_data": [{examples...}],
        "metrics": {...},
        "validation_report": {...}
    }
    """
```

---

## Reference Documents Now In Memory

The following documents have been read and are kept in memory for continuous reference during implementation:

| Document | Purpose | Key Sections |
|----------|---------|--------------|
| **REQUIREMENTS.md** | Most specific project requirements | Sections 1-12 (entire document) |
| **requriment_document.md** | Detailed training/packaging architecture | Sections 1-13 (entire document) |
| **DATASET_PREPARATION_GUIDE.md** | Data format and structure | Sections 1-3 (formats), Section 4 (dataset sizes) |
| **ORACLE_EBS_INTEGRATION_GUIDE.md** | Oracle EBS-specific patterns | Sections 1-2 (EBS specifics, SQL patterns) |
| **COMPLETE_END_TO_END_GUIDE.md** | Inference and critic loop | Sections 1-2 (inference, critic loop) |
| **sft_oracle_llama70b_lora.py** | Training entrypoint code | Lines 92-115 (label masking logic) |

### When Implementing, Refer To:
- **Prompt template:** REQUIREMENTS.md section 7 + requriment_document.md section 6.3
- **Label masking logic:** sft_oracle_llama70b_lora.py lines 95-115
- **Data validation:** DATASET_PREPARATION_GUIDE.md section 3
- **SQL patterns:** ORACLE_EBS_INTEGRATION_GUIDE.md section 1
- **Dataset sizes:** DATASET_PREPARATION_GUIDE.md section 4

### Step 2: Configure Data Pipeline

**File:** `data/oracle_sft_config.yaml`

```yaml
input:
  main_training: data/data_warehouse/oracle_train/training_data_sets_4jul2025_v5_askguru_compatible.jsonl
  corrections: data/data_warehouse/oracle_train/corrections.jsonl
  joins: data/data_warehouse/oracle_train/joins.json
  keywords: data/data_warehouse/oracle_train/keywords.json
  table_schemas: data/data_warehouse/oracle_train/*.json
  
output:
  base_dir: data/oracle_sft_conversations/
  full_dataset: oracle_sft_conversations_full.json
  train_dataset: oracle_sft_conversations_train.json
  val_dataset: oracle_sft_conversations_val.json
  test_dataset: oracle_sft_conversations_test.json
  quality_report: data_quality_report.json

processing:
  deduplication_enabled: true
  sql_normalization_enabled: true
  rag_context_generation: true
  quality_validation: true
  
splits:
  train_ratio: 0.80
  val_ratio: 0.10
  test_ratio: 0.10
  stratify_by:
    - num_tables
    - domain
    - has_joins
```

### Step 3: Update Training Script

**File:** `custom_oracle_llama/sft_oracle_llama70b_lora.py`

Modifications:
```python
# Add data loading logic
def load_training_data(config):
    """Load from pipeline output instead of raw"""
    dataset_path = config.oracle_sft_conversations_train
    # Load from json
    # Verify askGuru format
    return dataset

# Main training function
def train():
    # Load data from pipeline
    train_data = load_training_data(config)
    
    # Rest of training unchanged
    ...
```

---

## Quality Assurance Checklist

Before starting training, verify:

- [ ] **Deduplication:** ~100-200 duplicates removed (expected)
- [ ] **RAG Context:** 100% of examples have context
- [ ] **SQL Validation:** >99% parse successfully
- [ ] **Table Mapping:** All tables in SQL exist in schema files
- [ ] **Join Validation:** All joins documented and valid
- [ ] **Format:** All examples in correct askGuru format
- [ ] **Splits:** Train/val/test stratified correctly
- [ ] **Metadata:** All examples have complete metadata
- [ ] **Token Count:** Average prompt tokens within acceptable range (<6000)
- [ ] **SQL Quality:** Random spot-checks pass

---

## Risk Mitigation

### Risk 1: Duplicate Examples
**Impact:** Model trains on same patterns multiple times  
**Mitigation:** Automatic deduplication with reporting

### Risk 2: Poor RAG Context
**Impact:** Model doesn't learn table relationships properly  
**Mitigation:** Validate join coverage, log missing joins, manual review

### Risk 3: SQL Quality Variance
**Impact:** Model learns inconsistent formatting  
**Mitigation:** Normalize SQL, validate Oracle syntax

### Risk 4: Insufficient Joins Examples
**Impact:** Model underfits on complex multi-table queries  
**Mitigation:** Track % examples with joins, oversample if <20%

### Risk 5: Data Leakage
**Impact:** Val/test examples similar to train  
**Mitigation:** Stratified split, check for near-duplicates

---

## Timeline & Resources

| Phase | Task | Duration | Owner |
|-------|------|----------|-------|
| 1 | Data analysis & planning | ✅ Done | Zencoder |
| 2 | Data extraction & transformation | 2-3 hours | Dev |
| 3 | Format conversion | 1 hour | Dev |
| 4 | Quality assurance | 1-2 hours | Dev + QA |
| 5 | Train/val/test split | 30 min | Dev |
| - | **TOTAL** | ~5-7 hours | - |

---

## Approval Gate

Before implementing pipeline:

1. **Data Consolidation Strategy:** Approved ✅ (need confirmation)
   - Main training: 2600 → 2400-2500 after dedup
   - Corrections: 3 examples
   - Total: ~2403-2503 final examples

2. **RAG Context Approach:** Approved ✅ (need confirmation)
   - Use joins.json for relationship mapping
   - Use table schemas for column definitions
   - Use keywords.json for business mappings

3. **Quality Thresholds:** Approved ✅ (need confirmation)
   - >99% SQL parse success
   - 100% examples have RAG context
   - All tables verified in schemas

4. **Timeline:** Approved ✅ (need confirmation)
   - 5-7 hours to complete full pipeline

---

## Next Steps

1. **User Approval:** Confirm plan is acceptable
2. **Implement Pipeline:** Create data processing scripts
3. **Run Quality Checks:** Generate validation report
4. **Review Output:** Spot-check converted examples
5. **Approve Dataset:** Sign-off before training starts

---

## User Approval Needed: Critical Decisions

See **Critical Decision Points** section above (lines 599-639) for 5 key decisions:

1. **Schema Learning Examples (28 examples)** - Use for RAG context or training?
2. **Quality Tier Weighting** - Weight corrections 2-3x higher or equally?
3. **RAG Context Scope** - All tables, used only, or used + joined tables?
4. **Security Filters** - Always include or domain-aware?
5. **Data Augmentation** - Use all 2,500 as-is or implement augmentation?

### Recommended Approach (Pre-approved by analysis):
- ✅ Decision 1: Option A (use schema for context enrichment)
- ✅ Decision 2: Option B (weight corrections 2-3x)
- ✅ Decision 3: Option B (used + directly joined tables)
- ✅ Decision 4: Option B (domain-aware security filters)
- ✅ Decision 5: Option A (use all 2,500, augment if needed)

**Please confirm or suggest modifications to these recommendations.**

---

## Approval Sign-Off

**Plan Reviewed By:** Zencoder  
**Plan Approved By:** [User]  
**Date:** [To be filled]  
**Modifications:** [Any changes from plan]

