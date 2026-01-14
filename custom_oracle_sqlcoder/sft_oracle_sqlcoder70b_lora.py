"""
sft_oracle_sqlcoder70b_lora.py

Oracle EBS NL2SQL SFT entrypoint for SQLCoder-70B using askGuru-SQL components.
Secondary model for SQL-specialized fine-tuning (backup to LLaMA-3.1-70B).

SQLCoder-70B is CodeLlama-70B fine-tuned on SQL data.
Architecture: LlamaForCausalLM (same as LLaMA, compatible with askGuru framework)

Run from askGuru-SQL/ directory, e.g.:

accelerate launch --config_file train/config/zero3.yaml \
  custom_oracle_sqlcoder/sft_oracle_sqlcoder70b_lora.py \
  --model_name_or_path defog/sqlcoder-70b-alpha \
  --data_path data/oracle_sqlcoder_sft_train.json \
  --eval_data_path data/oracle_sqlcoder_sft_val.json \
  --output_dir outputs/oracle_sqlcoder70b_lora \
  --model_max_length 4096 \
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
from datasets import load_dataset, set_caching_enabled
from transformers import set_seed

from train.trainer.argument import ModelArguments, TrainingArguments, DataArguments, LoraArguments
from train.trainer.trainer import DeepCustomTrainer
from train.trainer.train_util import load_tokenizer_and_model, DataCollatorForGeneration
from custom_oracle_sqlcoder._preprocessing_sqlcoder import make_preprocess_fn_sqlcoder


def train() -> None:
    logging.basicConfig(level=logging.INFO)
    
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, LoraArguments))
    model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()

    if training_args.model_max_length is None:
        training_args.model_max_length = 4096
        logger.info("Set model_max_length to 4096 for SQLCoder (16K context window)")

    training_args.enable_dialect_router = bool(getattr(training_args, "enable_dialect_router", False))
    if training_args.enable_dialect_router:
        logger.warning("enable_dialect_router=True. SQLCoder is SQL-only, set to False.")

    # Path validation
    if not os.path.exists(data_args.data_path):
        raise FileNotFoundError(f"Training data not found: {data_args.data_path}")
    
    if getattr(data_args, "eval_data_path", None) and not os.path.exists(data_args.eval_data_path):
        logger.warning(f"Eval data path provided but not found: {data_args.eval_data_path}")
    
    set_seed(training_args.seed)
    logger.info(f"Seed set to {training_args.seed}")

    tokenizer, model = load_tokenizer_and_model(model_args, training_args, lora_args)
    logger.info(f"Loaded model: {model_args.model_name_or_path}")
    logger.info(f"Model architecture: {model.__class__.__name__}")

    set_caching_enabled(False)
    
    train_raw = load_dataset("json", data_files=data_args.data_path)["train"]
    logger.info(f"Loaded {len(train_raw)} training examples")
    
    eval_data = None
    if getattr(data_args, "eval_data_path", None):
        eval_raw = load_dataset("json", data_files=data_args.eval_data_path)["train"]
        logger.info(f"Loaded {len(eval_raw)} eval examples")
        eval_data = eval_raw
    else:
        logger.warning("No eval_data_path found; using 10% of train for eval")
        split = train_raw.train_test_split(test_size=0.1, seed=training_args.seed)
        train_raw = split["train"]
        eval_data = split["test"]
    
    train_data = train_raw
    logger.info(f"Final Train examples: {len(train_data)}, Eval examples: {len(eval_data) if eval_data else 0}")

    preprocess_fn_train = make_preprocess_fn_sqlcoder(tokenizer, training_args, is_eval=False)
    preprocess_fn_eval = make_preprocess_fn_sqlcoder(tokenizer, training_args, is_eval=True)

    train_data = train_data.map(
        preprocess_fn_train,
        remove_columns=train_data.column_names,
        desc="Preprocessing train data"
    )
    
    eval_data = eval_data.map(
        preprocess_fn_eval,
        remove_columns=eval_data.column_names,
        desc="Preprocessing eval data"
    )

    data_collator = DataCollatorForGeneration(
        tokenizer=tokenizer,
        model=model,
        padding="longest"
    )

    trainer = DeepCustomTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_data,
        eval_dataset=eval_data,
        peft_config=lora_args if training_args.use_lora else None
    )

    checkpoint = None
    if training_args.resume_from_checkpoint:
        checkpoint = training_args.resume_from_checkpoint
    
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    
    logger.info("Training completed")
    logger.info(f"Train loss: {train_result.training_loss}")
    
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    train()
