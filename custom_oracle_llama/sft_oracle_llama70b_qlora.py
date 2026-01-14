#!/usr/bin/env python3
"""
sft_oracle_llama70b_qlora.py (optional)

Same as sft_oracle_llama70b_lora.py but loads base model in 4-bit for QLoRA training.
Use only for experimentation or when GPUs are too limited for BF16 LoRA training.

Run from askGuru-SQL/ directory.

accelerate launch --config_file train/config/zero3.yaml \
  custom_oracle_llama/sft_oracle_llama70b_qlora.py \
  --model_name_or_path /models/llama-3.3-70b-instruct \
  --data_path data/oracle_sft_conversations.json \
  --output_dir outputs/oracle_llama70b_qlora \
  --model_max_length 8192
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
from datasets import load_dataset, set_caching_enabled
from transformers import set_seed

from train.trainer.argument import ModelArguments, TrainingArguments, DataArguments, LoraArguments
from train.trainer.trainer import DeepCustomTrainer
from train.trainer.train_util import DataCollatorForGeneration
from custom_oracle_llama.train_util_4bit import load_tokenizer_and_model_4bit
from custom_oracle_llama._preprocessing_utils import make_preprocess_fn


def train() -> None:
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting QLoRA training (4-bit quantization)")
    
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, LoraArguments))
    model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()

    if training_args.model_max_length is None:
        training_args.model_max_length = 8192
        logger.info("Set model_max_length to 8192 (default)")

    if not os.path.exists(data_args.data_path):
        raise FileNotFoundError(f"Training data not found: {data_args.data_path}")
    
    training_args.use_lora = True
    lora_args.q_lora = True
    logger.info("Enforced: use_lora=True, q_lora=True")

    set_seed(training_args.seed)
    logger.info(f"Seed set to {training_args.seed}")

    tokenizer, model = load_tokenizer_and_model_4bit(model_args, training_args, lora_args)
    logger.info(f"Loaded 4-bit model: {model_args.model_name_or_path}")

    preprocess_train = make_preprocess_fn(tokenizer, training_args, is_eval=False)

    set_caching_enabled(True)
    train_raw = load_dataset("json", data_files=data_args.data_path)["train"]
    logger.info(f"Loaded {len(train_raw)} training examples")
    
    train_ds = train_raw.map(
        preprocess_train,
        remove_columns=train_raw.column_names,
        num_proc=16,
        load_from_cache_file=True,
        desc="Tokenizing train dataset",
    )

    data_collator = DataCollatorForGeneration(tokenizer)
    trainer = DeepCustomTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=None,
        data_collator=data_collator,
    )
    
    trainer.train(resume_from_checkpoint=training_args.resume)
    trainer.save_state()
    trainer.save_model()
    logger.info(f"QLoRA training complete. Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    train()
