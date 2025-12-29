"""
train_util_4bit.py

Optional utilities to load a base model in 4-bit (bitsandbytes) for QLoRA training
WITHOUT editing askGuru-SQL/train/trainer/train_util.py.

This is only needed if you want QLoRA experiments. The recommended production path
in REQUIREMENTS.md uses LoRA BF16 training on 8×A100-80GB, then merge + quantize.

Usage pattern:
- Import load_tokenizer_and_model_4bit() from this module in a custom training entrypoint.
"""
from __future__ import annotations

import logging

import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

logger = logging.getLogger(__name__)


def _make_bnb_4bit_config():
    """Create BitsAndBytesConfig for 4-bit quantization."""
    try:
        from transformers import BitsAndBytesConfig
    except Exception as e:
        raise RuntimeError(
            "BitsAndBytesConfig is not available. Install transformers>=4.35 and bitsandbytes. "
            f"Original error: {e}"
        )
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def load_tokenizer_and_model_4bit(model_args, training_args, lora_args):
    """
    Load tokenizer and model in 4-bit for QLoRA training.
    
    This function mirrors askGuru's load_tokenizer_and_model() behavior but actually loads 4-bit weights.
    
    Args:
        model_args: ModelArguments from train.trainer.argument
        training_args: TrainingArguments from train.trainer.argument
        lora_args: LoraArguments from train.trainer.argument
    
    Returns:
        (tokenizer, model) tuple with model loaded in 4-bit
    """
    logger.info(f"Loading 4-bit model from {model_args.model_name_or_path}")
    
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        trust_remote_code=True,
        use_fast=False,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Set pad_token to eos_token ({tokenizer.eos_token})")

    bnb_cfg = _make_bnb_4bit_config()
    logger.info("Using 4-bit quantization (nf4, double_quant=True, bfloat16)")

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
        quantization_config=bnb_cfg,
        device_map="auto",
    )

    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=training_args.gradient_checkpointing
    )
    logger.info("Model prepared for k-bit training")

    if getattr(training_args, "use_lora", False):
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        logger.info(f"LoRA applied: r={lora_args.lora_r}, alpha={lora_args.lora_alpha}")
        
        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()
            logger.info("Enabled input_require_grads for gradient checkpointing")

    return tokenizer, model
