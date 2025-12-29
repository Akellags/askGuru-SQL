#!/usr/bin/env python3
"""
package_oracle_model.py

Packaging pipeline:
1) Merge LoRA adapter into base model -> merged HF checkpoint
2) Quantize merged checkpoint to 4-bit for inference (AWQ preferred, GPTQ fallback)
3) Write a manifest.json describing provenance and quantization config

This script does NOT modify askGuru code; it reuses train/utils/adapter_merge.py for merging.

Notes (air-gapped):
- AWQ: typically via AutoAWQ (package name varies: autoawq / awq)
- GPTQ: typically via auto-gptq
Your environment must provide one of these offline.

Examples:
python custom_oracle_llama/package_oracle_model.py \
  --base_model /models/llama-3.1-70b-instruct \
  --lora_adapter outputs/oracle_llama70b_lora \
  --merged_out outputs/merged_oracle_llama70b \
  --quant_out outputs/merged_oracle_llama70b_awq4 \
  --quant_method awq
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
from datetime import datetime
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)


def _sha256_of_file(path: str) -> Optional[str]:
    """Compute SHA256 hash of a file."""
    if not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: str, obj: Any) -> None:
    """Write JSON to file with directory creation."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def merge_lora(base_model: str, lora_adapter: str, merged_out: str) -> None:
    """Merge LoRA adapter into base model using askGuru's utility."""
    from train.utils.adapter_merge import apply_lora

    if os.path.exists(merged_out) and os.listdir(merged_out):
        logger.info(f"merged_out already exists and is non-empty: {merged_out}")
        return

    os.makedirs(merged_out, exist_ok=True)
    logger.info(f"Merging {lora_adapter} into {base_model}")
    apply_lora(model_name_or_path=base_model, output_path=merged_out, lora_path=lora_adapter)
    logger.info(f"Merge complete: {merged_out}")


def quantize_awq(merged_model: str, quant_out: str, group_size: int = 128, w_bit: int = 4) -> Dict[str, Any]:
    """
    Quantize using AutoAWQ.
    
    This implementation uses a best-effort API compatible with common AutoAWQ versions.
    If your internal AWQ package differs, adapt this function.
    """
    try:
        from awq import AutoAWQForCausalLM
    except Exception:
        try:
            from autoawq import AutoAWQForCausalLM
        except Exception as e:
            raise RuntimeError(f"AWQ quantization requested but AutoAWQ is not installed: {e}")

    from transformers import AutoTokenizer

    os.makedirs(quant_out, exist_ok=True)
    logger.info(f"Loading tokenizer from {merged_model}")
    
    tokenizer = AutoTokenizer.from_pretrained(merged_model, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading model for AWQ quantization (w_bit={w_bit}, group_size={group_size})")
    model = AutoAWQForCausalLM.from_pretrained(merged_model, trust_remote_code=True, low_cpu_mem_usage=True)
    quant_config = {"w_bit": w_bit, "q_group_size": group_size, "zero_point": True, "version": "GEMM"}

    logger.info("Quantizing model...")
    model.quantize(tokenizer, quant_config=quant_config)

    logger.info(f"Saving quantized model to {quant_out}")
    model.save_quantized(quant_out)
    tokenizer.save_pretrained(quant_out)

    return {"method": "awq", "w_bit": w_bit, "group_size": group_size, "quant_config": quant_config}


def quantize_gptq(merged_model: str, quant_out: str, w_bit: int = 4, group_size: int = 128) -> Dict[str, Any]:
    """
    Quantize using AutoGPTQ.
    
    Note: GPTQ typically needs calibration samples. Provide your own if required by your package.
    """
    try:
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    except Exception as e:
        raise RuntimeError(f"GPTQ quantization requested but auto-gptq is not installed: {e}")

    from transformers import AutoTokenizer

    os.makedirs(quant_out, exist_ok=True)
    logger.info(f"Loading tokenizer from {merged_model}")
    
    tokenizer = AutoTokenizer.from_pretrained(merged_model, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantize_config = BaseQuantizeConfig(bits=w_bit, group_size=group_size, damp_percent=0.01, desc_act=False)

    from custom_oracle_llama._calibration_data import get_calibration_texts
    calib_texts = get_calibration_texts(num_samples=8)

    logger.info(f"Loading model for GPTQ quantization (w_bit={w_bit}, group_size={group_size})")
    model = AutoGPTQForCausalLM.from_pretrained(
        merged_model,
        quantize_config=quantize_config,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    
    logger.info("Quantizing model with calibration samples...")
    model.quantize(calib_texts, use_triton=False)
    
    logger.info(f"Saving quantized model to {quant_out}")
    model.save_quantized(quant_out)
    tokenizer.save_pretrained(quant_out)

    return {"method": "gptq", "w_bit": w_bit, "group_size": group_size, "quantize_config": quantize_config.to_dict()}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    
    ap = argparse.ArgumentParser(description="Package Oracle model: merge LoRA + quantize")
    ap.add_argument("--base_model", required=True, help="Base model path (Hugging Face)")
    ap.add_argument("--lora_adapter", required=True, help="LoRA adapter directory path")
    ap.add_argument("--merged_out", required=True, help="Output directory for merged model")
    ap.add_argument("--quant_out", required=True, help="Output directory for quantized model")
    ap.add_argument("--quant_method", choices=["awq", "gptq", "none"], default="awq", help="Quantization method")
    ap.add_argument("--group_size", type=int, default=128, help="Quantization group size")
    ap.add_argument("--w_bit", type=int, default=4, help="Weight bit width")
    ap.add_argument("--manifest_path", default=None, help="Output path for manifest.json")
    ap.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    args = ap.parse_args()

    if not os.path.exists(args.base_model):
        raise FileNotFoundError(f"Base model not found: {args.base_model}")
    if not os.path.exists(args.lora_adapter):
        raise FileNotFoundError(f"LoRA adapter not found: {args.lora_adapter}")
    
    logger.info(f"Base model: {args.base_model}")
    logger.info(f"LoRA adapter: {args.lora_adapter}")

    if args.force:
        for p in [args.merged_out, args.quant_out]:
            if os.path.exists(p):
                logger.info(f"Removing existing {p}")
                shutil.rmtree(p, ignore_errors=True)

    logger.info("=" * 60)
    logger.info("STEP 1: Merge LoRA -> merged model")
    logger.info("=" * 60)
    merge_lora(args.base_model, args.lora_adapter, args.merged_out)

    quant_info: Dict[str, Any] = {"method": "none"}
    if args.quant_method != "none":
        logger.info("=" * 60)
        logger.info(f"STEP 2: Quantize merged model -> {args.quant_method}")
        logger.info("=" * 60)
        if args.quant_method == "awq":
            quant_info = quantize_awq(args.merged_out, args.quant_out, group_size=args.group_size, w_bit=args.w_bit)
        else:
            quant_info = quantize_gptq(args.merged_out, args.quant_out, w_bit=args.w_bit, group_size=args.group_size)
    else:
        logger.info("Quantization skipped (quant_method=none)")
        os.makedirs(args.quant_out, exist_ok=True)

    manifest = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "base_model": args.base_model,
        "lora_adapter": args.lora_adapter,
        "merged_out": args.merged_out,
        "quant_out": args.quant_out,
        "quantization": quant_info,
        "notes": "Merge+quant pipeline for Oracle NL2SQL. For best quant quality, calibrate with domain prompts.",
    }

    manifest_path = args.manifest_path or os.path.join(args.quant_out, "manifest.json")
    _write_json(manifest_path, manifest)
    logger.info(f"[OK] Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
