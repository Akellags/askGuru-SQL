#!/usr/bin/env python3
"""
package_oracle_model.py

Packaging pipeline:
1) Merge LoRA adapter into base model -> merged HF checkpoint
2) Quantize merged checkpoint to 4-bit for inference (AWQ preferred, GPTQ fallback)
3) Write a manifest.json describing provenance and quantization config

This script does NOT modify XiYan code; it reuses train/utils/adapter_merge.py for merging.

Notes (air-gapped):
- AWQ: typically via AutoAWQ (package name varies: autoawq / awq)
- GPTQ: typically via auto-gptq
Your environment must provide one of these offline.

Examples:
python custom_oracle_llama/package_oracle_model.py \
  --base_model /models/llama-3.3-70b-instruct \
  --lora_adapter outputs/oracle_llama70b_lora \
  --merged_out outputs/merged_oracle_llama70b \
  --quant_out outputs/merged_oracle_llama70b_awq4 \
  --quant_method awq

"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime
from typing import Any, Dict, Optional

import torch


def _sha256_of_file(path: str) -> Optional[str]:
    if not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def merge_lora(base_model: str, lora_adapter: str, merged_out: str) -> None:
    # Reuse XiYan's merge helper
    from train.utils.adapter_merge import apply_lora

    if os.path.exists(merged_out) and os.listdir(merged_out):
        print(f"[INFO] merged_out already exists and is non-empty: {merged_out}")
        return

    os.makedirs(merged_out, exist_ok=True)
    apply_lora(model_name_or_path=base_model, output_path=merged_out, lora_path=lora_adapter)


def quantize_awq(merged_model: str, quant_out: str, group_size: int = 128, w_bit: int = 4) -> Dict[str, Any]:
    """
    Quantize using AutoAWQ if available.

    This implementation uses a best-effort API compatible with common AutoAWQ versions.
    If your internal AWQ package differs, adapt this function.
    """
    try:
        from awq import AutoAWQForCausalLM  # some envs
    except Exception:
        try:
            from autoawq import AutoAWQForCausalLM  # other envs
        except Exception as e:
            raise RuntimeError(f"AWQ quantization requested but AutoAWQ is not installed: {e}")

    from transformers import AutoTokenizer

    os.makedirs(quant_out, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(merged_model, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoAWQForCausalLM.from_pretrained(merged_model, trust_remote_code=True, low_cpu_mem_usage=True)
    quant_config = {"w_bit": w_bit, "q_group_size": group_size, "zero_point": True, "version": "GEMM"}

    # Some AutoAWQ versions need calibration data; for NL2SQL we can use a small internal set.
    # If your AutoAWQ requires calibration, pass 'calib_data' here.
    model.quantize(tokenizer, quant_config=quant_config)

    model.save_quantized(quant_out)
    tokenizer.save_pretrained(quant_out)

    return {"method": "awq", "w_bit": w_bit, "group_size": group_size, "quant_config": quant_config}


def quantize_gptq(merged_model: str, quant_out: str, w_bit: int = 4, group_size: int = 128) -> Dict[str, Any]:
    """
    Quantize using AutoGPTQ if available.
    Note: GPTQ typically needs calibration samples. Provide your own if required by your package.
    """
    try:
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    except Exception as e:
        raise RuntimeError(f"GPTQ quantization requested but auto-gptq is not installed: {e}")

    from transformers import AutoTokenizer

    os.makedirs(quant_out, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(merged_model, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantize_config = BaseQuantizeConfig(bits=w_bit, group_size=group_size, damp_percent=0.01, desc_act=False)

    # Calibration: many installs require a list of texts. Use your own internal calibration set.
    # Here we provide a placeholder minimal sample. Replace with domain prompts for best quality.
    calib_texts = [
        "Return ONLY an Oracle SQL statement to answer the question based on the provided context.",
        "List the top 10 suppliers by total invoice amount in the last fiscal year.",
    ]

    model = AutoGPTQForCausalLM.from_pretrained(
        merged_model,
        quantize_config=quantize_config,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.quantize(calib_texts, use_triton=False)
    model.save_quantized(quant_out)
    tokenizer.save_pretrained(quant_out)

    return {"method": "gptq", "w_bit": w_bit, "group_size": group_size, "quant_config": quantize_config.to_dict()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--lora_adapter", required=True)
    ap.add_argument("--merged_out", required=True)
    ap.add_argument("--quant_out", required=True)
    ap.add_argument("--quant_method", choices=["awq", "gptq", "none"], default="awq")
    ap.add_argument("--group_size", type=int, default=128)
    ap.add_argument("--w_bit", type=int, default=4)
    ap.add_argument("--manifest_path", default=None)
    ap.add_argument("--force", action="store_true", help="Overwrite existing outputs.")
    args = ap.parse_args()

    if args.force:
        for p in [args.merged_out, args.quant_out]:
            if os.path.exists(p):
                shutil.rmtree(p, ignore_errors=True)

    print("[STEP] Merge LoRA -> merged model")
    merge_lora(args.base_model, args.lora_adapter, args.merged_out)

    quant_info: Dict[str, Any] = {"method": "none"}
    if args.quant_method != "none":
        print(f"[STEP] Quantize merged model -> {args.quant_method}")
        if args.quant_method == "awq":
            quant_info = quantize_awq(args.merged_out, args.quant_out, group_size=args.group_size, w_bit=args.w_bit)
        else:
            quant_info = quantize_gptq(args.merged_out, args.quant_out, w_bit=args.w_bit, group_size=args.group_size)
    else:
        print("[STEP] Quantization skipped")
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
    print(f"[OK] Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
