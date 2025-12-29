#!/usr/bin/env python3
"""
sft_oracle_llama70b_qlora.py (optional)

Same as sft_oracle_llama70b_lora.py but loads base model in 4-bit for QLoRA training.
Use only for experimentation or when GPUs are too limited for BF16 LoRA training.

Run from XiYan-SQLTraining/ directory.
"""
from __future__ import annotations

import os
import sys
import json
from typing import Any, Dict

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

IGNORE_TOKEN_ID = -100


def train() -> None:
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, LoraArguments))
    model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()

    # Force QLoRA
    training_args.use_lora = True
    lora_args.q_lora = True

    set_seed(training_args.seed)
    tokenizer, model = load_tokenizer_and_model_4bit(model_args, training_args, lora_args)

    def preprocess(example: Dict[str, Any]) -> Dict[str, Any]:
        conversations = example["conversations"]
        text = tokenizer.apply_chat_template(conversations, tokenize=False, add_generation_prompt=False)
        enc = tokenizer(text, truncation=True, max_length=training_args.model_max_length)

        target = conversations[1]["content"]
        idx = text.find(target)
        target_idx = enc.char_to_token(idx) if idx >= 0 else None
        labels = enc["input_ids"].copy()
        if target_idx is not None:
            labels[:target_idx] = [IGNORE_TOKEN_ID] * target_idx
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": labels}

    set_caching_enabled(True)
    train_raw = load_dataset("json", data_files=data_args.data_path)["train"]
    train_ds = train_raw.map(preprocess, remove_columns=train_raw.column_names, num_proc=16, load_from_cache_file=True)

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


if __name__ == "__main__":
    train()
