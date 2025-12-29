# Dataset Preparation Guide for askGuru-SQL + Mistral-Small-3.2-24B

## 📊 Available Training Data

### Current Datasets in Codebase

| File | Location | Size | Count | Format | Purpose |
|------|----------|------|-------|--------|---------|
| **train_examples.json** | `train/datasets/` | 0.70 MB | 221 samples | Assembled (formatted for training) | Direct model training input |
| **train.json** (raw) | `data/data_warehouse/bird_train/raw_data/` | 103.81 KB | 221 samples | Raw format | Source data before processing |

### Data Source
- **Dataset Name**: Bird Training Dataset (Movie Platform)
- **Domain**: Movie database queries
- **Dialect**: SQLite (`nl2sqlite`)
- **Language**: Chinese (prompts), with English questions
- **Example DB**: `movie_platform` (tables: movies, ratings, lists, users)

---

## 🏗️ Data Structures

### 1. Raw Format (Source Data)
```json
{
  "db_id": "movie_platform",
  "question": "Name movie titles released in year 1945...",
  "evidence": "released in the year 1945 refers to movie_release_year = 1945;",
  "SQL": "SELECT movie_title FROM movies WHERE..."
}
```

**Required Fields**:
- `db_id` or `db_name`: Database identifier
- `question`: Natural language query
- `SQL`: Ground-truth SQL query
- `evidence` (optional): Explanation of mapping between NL and SQL

---

### 2. Processed Format (After data_processing.py)
```json
{
  "idx": 0,
  "db_name": "movie_platform",
  "question": "Name movie titles released in year 1945...",
  "evidence": "released in the year 1945 refers to movie_release_year = 1945;",
  "db_schema": "# M-Schema Format (generated from database)\n# Table: movies\n[\n  (movie_id:INTEGER, Primary Key, Examples: [1, 2, 3]),\n  (movie_title:TEXT, Examples: [La Antena, Elementary Particles]),\n  ...\n]",
  "sql": "SELECT movie_title FROM movies WHERE..."
}
```

**Key Addition**: 
- `db_schema`: M-Schema representation generated from live database (or provided)

---

### 3. Training Format (After data_assembler.py)
```json
{
  "id": 0,
  "conversations": [
    {
      "role": "user",
      "content": "You are a SQLite expert，现在need to阅读并理解下面的【database schema】...\n[User Question]\nName movie titles released in year 1945...\n【database schema】\n...\n[Reference Information]\n...\n```sql"
    },
    {
      "role": "assistant",
      "content": "SELECT movie_title FROM movies WHERE movie_release_year = 1945 ORDER BY movie_popularity DESC"
    }
  ],
  "sql_type": "nl2sqlite"
}
```

**This is what training uses**: Instruction-following format with system prompts

---

## 📋 Database Schema Details (M-Schema Format)

### What Gets Generated
The framework generates "M-Schema" - an enhanced schema representation including:

1. **Table Information**
   - Table name
   - All columns with types
   - Primary keys
   - Comments/descriptions
   - Example values for each column

2. **Column Information per Field**
   - Column name
   - Data type (INTEGER, TEXT, FLOAT, etc.)
   - Primary key flag
   - Nullable flag
   - Default value
   - Examples (sample values from DB)
   - Comments
   - Category (metadata)

3. **Relationships**
   - Foreign key mappings
   - Table relationships

---

## 📦 Recommended Dataset Size for Training Mistral-Small-3.2-24B

### Guidelines by Model Size

| Model Size | Min Samples | Recommended | Optimal |
|-----------|------------|------------|---------|
| **7B-14B** | 500 | 2,000-5,000 | 10,000-50,000 |
| **24B** | 1,000 | 5,000-10,000 | **20,000-100,000** |
| **32B+** | 2,000 | 10,000-20,000 | 50,000-200,000+ |

### For Mistral-Small-3.2-24B:
- **Minimum**: 1,000 samples
- **Good**: 5,000-10,000 samples
- **Excellent**: 20,000-50,000 samples
- **Very Good**: 50,000+ samples with diversity

### Current Codebase
- **Available**: 221 samples
- **Status**: **Too small for effective finetuning** of a 24B model
- **Recommendation**: Expand to **at least 5,000-10,000 samples**

---

This guide provides comprehensive information about dataset preparation and structure requirements.