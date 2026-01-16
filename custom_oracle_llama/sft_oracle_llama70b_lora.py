#!/usr/bin/env python3
"""
sft_oracle_llama70b_lora.py

Oracle EBS NL2SQL SFT entrypoint for LLaMA-3.3-70B using askGuru-SQL components,
WITHOUT modifying existing askGuru code.

This script is intentionally close to train/sft4askguru.py but:
- defaults to Oracle-only (dialect router off)
- intended for LoRA BF16 training on 8×A100-80GB
- keeps output contract as "SQL only" (enforced by dataset; optional lightweight checks)

Run from askGuru-SQL/ directory, e.g.

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

import logging
import os
import sys
from typing import Any, Dict

logger = logging.getLogger(__name__)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import transformers
from datasets import load_dataset
from transformers import set_seed

from train.trainer.argument import ModelArguments, TrainingArguments, DataArguments, LoraArguments
from train.trainer.trainer import DeepCustomTrainer
from train.trainer.train_util import load_tokenizer_and_model, DataCollatorForGeneration
from custom_oracle_llama._preprocessing_utils import make_preprocess_fn


def train() -> None:
    logging.basicConfig(level=logging.INFO)
    
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, LoraArguments))
    model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()

    if training_args.model_max_length is None:
        training_args.model_max_length = 8192
        logger.info("Set model_max_length to 8192 (default)")

    training_args.enable_dialect_router = bool(getattr(training_args, "enable_dialect_router", False))
    if training_args.enable_dialect_router:
        logger.warning("enable_dialect_router=True. For Oracle-only training, set it to False.")

    if not os.path.exists(data_args.data_path):
        raise FileNotFoundError(f"Training data not found: {data_args.data_path}")
    
    set_seed(training_args.seed)
    logger.info(f"Seed set to {training_args.seed}")

    tokenizer, model = load_tokenizer_and_model(model_args, training_args, lora_args)
    logger.info(f"Loaded model: {model_args.model_name_or_path}")

    preprocess_train = make_preprocess_fn(tokenizer, training_args, is_eval=False)
    preprocess_eval = make_preprocess_fn(tokenizer, training_args, is_eval=True)
    
    train_raw = load_dataset("json", data_files=data_args.data_path)["train"]
    logger.info(f"Loaded {len(train_raw)} training examples")
    
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
        logger.info(f"Loaded {len(eval_raw)} eval examples")

        eval_ds = eval_raw.map(
            preprocess_eval,
            remove_columns=eval_raw.column_names,
            num_proc=16,
            load_from_cache_file=True,
            desc="Tokenizing eval dataset",
        )

    data_collator = DataCollatorForGeneration(tokenizer)

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
    logger.info(f"Training complete. Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    train()
