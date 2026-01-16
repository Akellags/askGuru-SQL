#!/usr/bin/env python3
"""
sft_oracle_llama70b_lora.py

Oracle EBS NL2SQL SFT entrypoint for LLaMA-3.3-70B-Instruct using XiYan-SQLTraining components,
WITHOUT modifying existing XiYan code.

This script is intentionally close to train/sft4xiyan.py but:
- defaults to Oracle-only (dialect router off)
- intended for LoRA BF16 training on 8×A100-80GB
- keeps output contract as "SQL only" (enforced by dataset; optional lightweight checks)

Run from XiYan-SQLTraining/ directory, e.g.

accelerate launch --config_file train/config/zero3.yaml \
  custom_oracle_llama/sft_oracle_llama70b_lora.py \
  --model_name_or_path /models/llama-3.3-70b-instruct \
  --data_path data/oracle_sft_conversations.json \
  --output_dir outputs/oracle_llama70b_lora \
  --model_max_length 8192 \
  --use_lora True \
  --q_lora False \
  --enable_dialect_router False
"""
from __future__ import annotations

import os
import sys
import json
from typing import Any, Dict, List

# Ensure XiYan modules are importable when running from XiYan-SQLTraining root.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import transformers
from datasets import load_dataset, set_caching_enabled
from transformers import set_seed

from train.trainer.argument import ModelArguments, TrainingArguments, DataArguments, LoraArguments
from train.trainer.trainer import DeepCustomTrainer
from train.trainer.train_util import load_tokenizer_and_model, DataCollatorForGeneration

IGNORE_TOKEN_ID = -100


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _basic_sql_only_check(sql: str) -> bool:
    """Lightweight guard to catch obviously non-SQL outputs in training data."""
    if not sql:
        return False
    s = sql.strip()
    # Allow WITH / SELECT only for this project; adjust if you allow DML.
    if s[:6].upper() not in ("SELECT", "WITH  ", "WITH\n", "WITH\t"):
        # Many SQL start with WITH; "WITH  " covers typical
        # If it starts with "WITH", that's fine.
        if not s.upper().startswith("WITH"):
            return False
    # Block markdown fences
    if "```" in s:
        return False
    return True


def train() -> None:
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, LoraArguments))
    model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()

    # Recommended defaults (can be overridden by CLI)
    if training_args.model_max_length is None:
        training_args.model_max_length = 8192

    # Oracle-only: keep dialect router off by default
    training_args.enable_dialect_router = bool(getattr(training_args, "enable_dialect_router", False))
    if training_args.enable_dialect_router:
        print("[WARN] enable_dialect_router=True. For Oracle-only training, set it to False.")

    # Set seed
    set_seed(training_args.seed)

    # Load model/tokenizer using XiYan utility (LoRA path)
    tokenizer, model = load_tokenizer_and_model(model_args, training_args, lora_args)

    def preprocess_train(example: Dict[str, Any]) -> Dict[str, Any]:
        conversations = example["conversations"]
        # Expect: [user, assistant]
        text = tokenizer.apply_chat_template(conversations, tokenize=False, add_generation_prompt=False)
        enc = tokenizer(text, truncation=True, max_length=training_args.model_max_length)

        target = conversations[1]["content"]
        if not _basic_sql_only_check(target):
            # keep but flag; in strict mode you'd drop it
            example_id = example.get("id", "<no-id>")
            print(f"[WARN] Non SQL-only target detected for {example_id}")

        idx = text.find(target)
        target_idx = enc.char_to_token(idx) if idx >= 0 else None
        labels = enc["input_ids"].copy()

        if target_idx is not None:
            labels[:target_idx] = [IGNORE_TOKEN_ID] * target_idx
        else:
            # If alignment fails, train on all tokens (worst case); better: drop the row.
            # Here we keep but warn.
            example_id = example.get("id", "<no-id>")
            print(f"[WARN] Failed to align target span for {example_id}; training on full sequence.")
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": labels}

    set_caching_enabled(True)
    train_raw = load_dataset("json", data_files=data_args.data_path)["train"]
    train_ds = train_raw.map(
        preprocess_train,
        remove_columns=train_raw.column_names,
        num_proc=16,
        load_from_cache_file=True,
        desc="Tokenizing train dataset",
    )

    eval_ds = None
    if getattr(data_args, "eval_data_path", None):
        eval_raw = load_dataset("json", data_files=data_args.eval_data_path)["train"]

        def preprocess_eval(example: Dict[str, Any]) -> Dict[str, Any]:
            conversations = example["conversations"]
            # Provide generation prompt: set add_generation_prompt=True so model generates assistant content.
            # For LLaMA chat templates, this is usually correct.
            prompt = tokenizer.apply_chat_template(conversations[:1], tokenize=False, add_generation_prompt=True)
            enc = tokenizer(prompt, truncation=True, max_length=training_args.model_max_length)
            label_sql = conversations[1]["content"]
            return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": label_sql}

        eval_ds = eval_raw.map(
            preprocess_eval,
            remove_columns=eval_raw.column_names,
            num_proc=16,
            load_from_cache_file=True,
            desc="Tokenizing eval dataset",
        )

    data_collator = DataCollatorForGeneration(tokenizer)

    # Use XiYan's DeepCustomTrainer (supports deepspeed configs used in this repo)
    trainer = DeepCustomTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
    )

    trainer.train(resume_from_checkpoint=training_args.resume)
    trainer.save_state()
    trainer.save_model()


if __name__ == "__main__":
    train()
