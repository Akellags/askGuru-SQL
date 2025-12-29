#!/usr/bin/env python3
"""
build_oracle_sft_dataset.py

Convert raw Oracle EBS NL2SQL examples into XiYan-SQLTraining SFT JSON format:

[
  {
    "id": "...",
    "sql_type": "oracle",
    "conversations": [
      {"role": "user", "content": "<prompt with rules + rag_context + question>"},
      {"role": "assistant", "content": "<ORACLE SQL ONLY>"}
    ],
    "meta": {...}
  },
  ...
]

Input formats supported:
1) JSONL: each line is an object with at least:
   - question (str)
   - sql_gold (str)  (Oracle SQL)
   - rag_context (str)
   Optional: domain, difficulty, tags, etc.

2) JSON: array of such objects.

Best practice: make rag_context already pruned (top-K tables/columns/join paths).
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_RULES = """You are a Text-to-SQL generator.
Return ONLY a single Oracle SQL statement.
- Do NOT use markdown.
- Do NOT add explanations.
- Use Oracle SQL dialect.
- Prefer explicit JOIN syntax.
- If the question is ambiguous, make the safest reasonable assumption based on provided context.
"""


def _read_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        return []
    # Heuristic: JSON array vs JSONL
    if content[0] == "[":
        return json.loads(content)
    items: List[Dict[str, Any]] = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        items.append(json.loads(line))
    return items


def _write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _normalize_sql(sql: str) -> str:
    sql = (sql or "").strip()
    # Keep SQL as-is, but remove trailing markdown fences if present
    if sql.startswith("```"):
        sql = sql.strip("`").strip()
    return sql


def _build_user_prompt(
    question: str,
    rag_context: str,
    rules: str = DEFAULT_RULES,
    extra_instructions: Optional[str] = None,
) -> str:
    # Structured prompt sections help the model and reduce wasted tokens.
    parts: List[str] = []
    parts.append(rules.strip())
    if extra_instructions:
        parts.append(str(extra_instructions).strip())

    parts.append("# Context (retrieved)\n" + (rag_context or "").strip())
    parts.append("# Question\n" + (question or "").strip())
    return "\n\n".join([p for p in parts if p])


def convert_examples(
    raw: List[Dict[str, Any]],
    dataset_id: str,
    rules: str,
    extra_instructions: Optional[str] = None,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, ex in enumerate(raw):
        question = ex.get("question") or ex.get("nl_question") or ex.get("utterance")
        sql = ex.get("sql_gold") or ex.get("sql") or ex.get("gold_sql")
        rag_context = ex.get("rag_context") or ex.get("context") or ""

        if not question or not sql:
            # Skip incomplete rows, but keep track via stderr in real pipelines.
            continue

        user_prompt = _build_user_prompt(
            question=str(question),
            rag_context=str(rag_context),
            rules=rules,
            extra_instructions=extra_instructions,
        )
        sql_norm = _normalize_sql(str(sql))

        item: Dict[str, Any] = {
            "id": ex.get("id") or f"{dataset_id}_{i:06d}",
            "sql_type": "oracle",
            "conversations": [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": sql_norm},
            ],
            "meta": {
                "dataset_id": dataset_id,
                "domain": ex.get("domain"),
                "difficulty": ex.get("difficulty"),
                "tags": ex.get("tags"),
                "source": ex.get("source"),
            },
        }
        out.append(item)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_raw", required=True, help="Path to raw JSON or JSONL examples.")
    ap.add_argument("--output_sft", required=True, help="Output path for XiYan SFT JSON.")
    ap.add_argument("--dataset_id", required=True, help="A version id, e.g., oracle_ebs_v1")
    ap.add_argument("--rules_file", default=None, help="Optional path to a rules text file.")
    ap.add_argument("--extra_instructions", default=None, help="Extra text appended after rules.")
    args = ap.parse_args()

    rules = DEFAULT_RULES
    if args.rules_file:
        with open(args.rules_file, "r", encoding="utf-8") as f:
            rules = f.read()

    raw = _read_json_or_jsonl(args.input_raw)
    converted = convert_examples(
        raw=raw,
        dataset_id=args.dataset_id,
        rules=rules,
        extra_instructions=args.extra_instructions,
    )
    _write_json(args.output_sft, converted)
    print(f"Wrote {len(converted)} examples to {args.output_sft}")


if __name__ == "__main__":
    main()
