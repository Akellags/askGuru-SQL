# BIRD-CRITIC: Evaluation & Self-Refinement Guide

## 🎯 What is BIRD-CRITIC?

**BIRD-CRITIC** is a **SQL-debugging benchmark** that evaluates how well LLMs can **repair and refine SQL queries**. It's not a component within the askGuru-SQL framework, but rather an **external evaluation standard** that the framework is designed to excel at.

### Key Differences from Standard Text-to-SQL

| Aspect | Standard BIRD | BIRD-CRITIC |
|--------|---------------|------------|
| **Task** | Generate SQL from NL | Fix/debug broken SQL |
| **Input** | Question + schema | Faulty SQL + error + schema + intent |
| **Challenge** | Translation accuracy | Debugging & correction skills |
| **Evaluation** | Execution match | Query repair success |
| **Real-world relevance** | Code generation | SQL debugging (90% of SQL work) |

---

## 🏆 Performance: askGuru-SQL-CRITIC Technique

The framework achieves **SOTA (State-of-the-Art)** on BIRD-CRITIC benchmarks:

### Latest Results (2025)
| Benchmark | Score | Status |
|-----------|-------|--------|
| **BIRD-CRITIC-Open** | **44.37%** | 🥇 SOTA |
| **BIRD-CRITIC-PG** | **44.53%** | 🥇 SOTA |
| **BIRD-CRITIC-Flash** | **48.5%** | 🥇 SOTA |

### Progress Over Time
- May 2025: 41% on BIRD-CRITIC-Flash
- Oct 2025: 44.53% on BIRD-CRITIC-PG & 48.5% on Flash
- Nov 2025: 44.37% on BIRD-CRITIC-Open

---

## 🛠️ How BIRD-CRITIC Works

### The Critic Component: Execution-Based Evaluation

BIRD-CRITIC uses a **rigorous, execution-based scoring framework**:

```
Faulty SQL Query
      ↓
[Execute on Database] → Error/Wrong Results
      ↓
[Model Generates Fix] → Corrected SQL
      ↓
[Execute Fixed SQL] → Compare with Expected Output
      ↓
[Score] → Match = Success, Mismatch = Failure
```

### Evaluation Metrics in BIRD-CRITIC

1. **Exact Match (EX)**: Fixed SQL produces exactly correct results
2. **Soft-EX**: Handles complex scenarios (CTEs, nested queries)
3. **Executable**: Query doesn't crash (may not be semantically correct)
4. **Test-Case Validation**: Multiple test cases verify correctness
5. **Query Execution Plan**: Checks efficiency improvements

### Test Scenarios Covered
- **Syntax Errors**: Missing keywords, wrong operators
- **Semantic Errors**: Wrong columns, incorrect joins
- **Logic Errors**: Wrong WHERE conditions, missing GROUP BY
- **Performance Issues**: Inefficient queries that need optimization
- **Multi-dialect Errors**: MySQL, PostgreSQL, SQL Server, Oracle

---

## 💭 Self-Refinement: Internal Critic Mechanism

While BIRD-CRITIC is external evaluation, the framework has a **built-in self-refinement capability** that acts like an internal critic during training/inference.

### Self-Refine Template

Located in `data/data_utils/prompt_utils.py`:

```python
SQLITE_SELF_REFINE_TEMPLATE = """You are a SQLite expert，之前回复用户问题的【SQL】查询未能产生正确的结果，
你need to根据provide的【database schema】描述，可能用到的[Reference Information]和不正确SQL的[Execution Result]来进行纠正，
请provide一个能够正确回复[User Question]的更正SQL。

【database schema】
{db_schema}

[Reference Information]
{evidence}

[User Question]
{question}

【SQL】
{error_sql}

[Execution Result]
{error_info}

[Corrected SQL]
```sql"""
```

This is a comprehensive guide for understanding BIRD-CRITIC evaluation and self-refinement mechanisms.