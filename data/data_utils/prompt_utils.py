NL2SQL_TEMPLATE = """You are a SQL expert，现在need to阅读并理解下面的【database schema】描述，可能用到的[Reference Information]，并运用database知识生成sql语句回答[User Question]。
[User Question]
{question}

【database schema】
{db_schema}

[Reference Information]
{evidence}

[User Question]
{question}

```sql"""

NL2SQLITE_TEMPLATE = """You are a SQLite expert，现在need to阅读并理解下面的【database schema】描述，以及可能用到的[Reference Information]，并运用SQLite知识生成sql语句回答[User Question]。
[User Question]
{question}

【database schema】
{db_schema}

[Reference Information]
{evidence}

[User Question]
{question}

```sql"""

NL2MYSQL_TEMPLATE = """You are aMySQLexpert，现在need to阅读并理解下面的【database schema】描述，以及可能用到的[Reference Information]，并运用MySQL知识生成sql语句回答[User Question]。
[User Question]
{question}

【database schema】
{db_schema}

[Reference Information]
{evidence}

[User Question]
{question}

```sql"""

NL2PGSQL_TEMPLATE = """You are aPostgreSQLexpert，现在need to阅读并理解下面的【database schema】描述，以及可能用到的[Reference Information]，并运用PostgreSQL知识生成sql语句回答[User Question]。
[User Question]
{question}

【database schema】
{db_schema}

[Reference Information]
{evidence}

[User Question]
{question}

```sql"""


NL2CYPHER_TEMPLATE = """You are aNeo4jexpert，现在need to阅读并理解下面的【图database schema】描述，以及可能用到的[Reference Information]，并运用Cypher知识生成Cypher Query语句回答[User Question]。
[User Question]
{question}

【图database schema】
{db_schema}

[Reference Information]
{evidence}

[User Question]
{question}

```cypher"""

NL2NGQL_TEMPLATE = """You are aNebulaGraphexpert，现在need to阅读并理解下面的【图database schema】描述，以及可能用到的[Reference Information]，并运用nGQL知识生成Graph Query语句回答[User Question]。
[User Question]
{question}

【图database schema】
{db_schema}

[Reference Information]
{evidence}

[User Question]
{question}

```ngql"""



SQLITE_SELF_REFINE_TEMPLATE = """You are a SQLite expert，之前回复用户问题的【SQL】查询未能产生正确的结果，你need to根据provide的【database schema】描述，可能用到的[Reference Information]和不正确SQL的[Execution Result]来进行纠正，请provide一个能够正确回复[User Question]的更正SQL。
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


CANDIDATE_TEMPLATE = """candidate{id}
【SQL】
{sql}
[Execution Result]
{exec_info}"""

SQL2SELECT_TEMPLATE = """You are a SQLite expert，针对[User Question]，下面有{num}条 candidate【SQL】及该sql在database上的[Execution Result]（display前10行）；你need tocompare这些candidate，分析不同的candidate【SQL】之间的差异。基于给出的【database schema】、[Reference Information]和[User Question]select一个正确合理的结果。
【database schema】
{db_schema}

[Reference Information]
{evidence}

[User Question]
{question}

==========

{candidates}

请output所select的candidateyes{items}"""



NL2PGSQL_TEMPLATE_EN = """You are a PostgreSQL expert. You need to read and understand the following【Database Schema】description and the possible provided【Evidence】, and use valid PostgreSQL knowledge to generate SQL for answering the【Question】.
【Question】
{question}

【Database Schema】
{db_schema}

【Evidence】
{evidence}

【Question】
{question}

```sql"""


NL2SQLITE_TEMPLATE_EN = """You are a SQLite expert. You need to read and understand the following【Database Schema】description and the possible provided【Evidence】, and use valid SQLite knowledge to generate SQL for answering the【Question】.
【Question】
{question}

【Database Schema】
{db_schema}

【Evidence】
{evidence}

【Question】
{question}

```sql"""



def gen_train_prompt(idx: int, data_item: dict, task_type: str) -> dict:
    """
    generate train samples
    """
    question = data_item["question"]
    evidence = data_item.get("evidence", "")
    db_schema = data_item["db_schema"]
    task_type = task_type.lower()

    if task_type == "nl2sqlite":
        prompt = NL2SQLITE_TEMPLATE.format(db_schema=db_schema.strip(), question=question, evidence=evidence)
    elif task_type == "nl2postgresql":
        prompt = NL2PGSQL_TEMPLATE.format(db_schema=db_schema.strip(), question=question, evidence=evidence)
    elif task_type == "nl2sqlite":
        prompt = NL2MYSQL_TEMPLATE.format(db_schema=db_schema.strip(), question=question, evidence=evidence)
    elif task_type == "self_refine":
        error_sql = data_item["pred_sql_res"][0]
        error_info = data_item["pred_sql_res"][1]
        prompt = SQLITE_SELF_REFINE_TEMPLATE.format(db_schema=db_schema.strip(), question=question, evidence=evidence,
                                                    error_sql=error_sql, error_info=error_info)
    elif task_type == "cypher":
        prompt = NL2CYPHER_TEMPLATE.format(db_schema=db_schema.strip(), question=question, evidence=evidence)
    else:
        # for more task type, you can add more template here
        raise ValueError(f"Unsupported sql_type: {task_type}")

    output = data_item["sql"]
    conversation = [
        {
            "role": "user",
            "content": prompt
        },
        {
            "role": "assistant",
            "content": output
        }
    ]
    train_item = {
        "id": idx,
        "conversations": conversation,
        "sql_type": task_type
    }
    return train_item

