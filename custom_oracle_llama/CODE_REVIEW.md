# Code Review: custom_oracle_llama

**Date:** 2025-12-27  
**Status:** ✅ Moved to project root and refined

## Summary of Improvements Made

### 1. **build_oracle_sft_dataset.py**
✅ **Improvements:**
- Added proper logging with `logging` module (configurable level)
- Enhanced `convert_examples()` to return skip count for better tracking
- Improved error handling with warning messages for skipped examples
- Added comprehensive docstrings and type hints
- Added `log_level` CLI argument for debugging

⚠️ **Potential Issues:**
- Line 33: Using `tuple[List[Dict[str, Any]], int]` requires Python 3.9+. For compatibility with older Python versions, should use `Tuple[List[Dict[str, Any]], int]` from `typing`.

**Recommendation:** Update type hints for Python 3.8 compatibility if needed.

---

### 2. **sft_oracle_llama70b_lora.py**
✅ **Improvements:**
- Added structured logging with clear [INFO], [WARN] messages
- Simplified `_basic_sql_only_check()` logic (cleaner condition)
- Added logging at key points: model loading, dataset loading, training completion
- Better error context with example IDs in warnings
- Clean separation of concerns (preprocessing functions)

⚠️ **Potential Issues:**
- Line 38: Imports unused `numpy as np` (should remove)
- No validation that `data_args.data_path` exists before loading
- No handling of empty dataset case

**Recommendations:**
1. Remove unused import: `import numpy as np`
2. Add file existence check: `if not os.path.exists(data_args.data_path): raise FileNotFoundError(...)`
3. Handle case where dataset is empty after preprocessing

---

### 3. **package_oracle_model.py**
✅ **Improvements:**
- Comprehensive logging at each step (MERGE, QUANTIZE, MANIFEST)
- Clear separation of merge and quantize logic
- SHA256 utility (defined but not used currently - good for future manifest enhancement)
- Better error messages for missing quantization libraries
- Added more calibration examples for GPTQ (4 examples vs 2 before)
- Proper docstrings

⚠️ **Potential Issues:**
- Line 139: `quantize_config.to_dict()` - `BaseQuantizeConfig` may not have `to_dict()` method in all versions
- Missing validation that base model and adapter paths exist before attempting merge
- Manifest doesn't record base model/adapter hashes (SHA256 function defined but unused)

**Recommendations:**
1. Add path existence validation at start of `main()`
2. Wrap `quantize_config.to_dict()` in try-except or use `vars(quantize_config)` as fallback
3. Optional: Compute and store hashes in manifest for provenance tracking
4. Add progress indicators for long operations (model loading, quantization)

---

### 4. **train_util_4bit.py**
✅ **Improvements:**
- Added proper logging for all major steps
- Clear docstrings with parameter descriptions
- Good error handling for missing BitsAndBytesConfig

⚠️ **Potential Issues:**
- No validation of model loading success
- If tokenizer loading fails, error message could be more informative
- `device_map="auto"` might not be optimal for distributed training (ZeRO-3)

**Recommendations:**
1. Add try-except around model loading with informative error
2. Log tokenizer vocab size and special tokens
3. Consider accepting `device_map` as parameter for flexibility

---

### 5. **sft_oracle_llama70b_qlora.py**
✅ **Improvements:**
- Added logging and clear messages
- Proper structure matching LoRA training
- Good docstring explaining use case

⚠️ **Potential Issues:**
- Duplicated preprocessing logic from LoRA script (violates DRY principle)
- No eval dataset support (unlike LoRA version)
- Missing logging for skipped/aligned examples

**Recommendations:**
1. Extract preprocessing function to shared module to avoid duplication
2. Add optional eval dataset support
3. Add warnings for non-SQL outputs like LoRA version does

---

### 6. **sql_guardrail.py**
✅ **Improvements:**
- Comprehensive validation logic with multiple checks
- Modular design with `SQLGuardrail` class
- Good documentation and examples
- Proper handling of markdown, semicolons, multi-statement SQL
- Optional Oracle hook design

⚠️ **Potential Issues:**
- Regex `SQL_FENCE_RE` could be more robust (doesn't handle ````sql blocks properly in all cases)
- `build_retry_prompt()` doesn't validate that original_prompt is not empty
- Hook signature tuple unpacking assumes 3 return values (fragile)

**Recommendations:**
1. Improve markdown fence regex to handle ````sql notation
2. Add validation in `build_retry_prompt()`: `if not original_prompt: raise ValueError(...)`
3. Create named tuple or dataclass for hook return type for type safety
4. Add timeout parameter for Oracle parse hook (prevent hanging)

---

### 7. **inference/vllm_config.yaml**
✅ **Good State:**
- Clear documentation in comments
- Safe defaults with explicit tuning guidance
- Temperature=0.0 enforces deterministic output (correct choice)

⚠️ **Minor Issue:**
- Path `outputs/merged_oracle_llama70b_awq4` is hardcoded (should be parameterized)

**Recommendation:**
Add note that users should update model path before deployment.

---

### 8. **README.md**
✅ **Excellent:**
- Comprehensive workflow documentation
- Clear step-by-step commands
- Good troubleshooting section
- Design principles well explained

---

## High-Priority Issues

| Issue | File | Severity | Fix |
|-------|------|----------|-----|
| Unused import | `sft_oracle_llama70b_lora.py:38` | Low | Remove `import numpy as np` |
| Type hint Python 3.8 compat | `build_oracle_sft_dataset.py:109` | Medium | Use `Tuple` from `typing` instead of `tuple[...]` |
| Path validation missing | `package_oracle_model.py` | Medium | Add existence check for base_model and lora_adapter |
| Regex might not handle all markdown | `sql_guardrail.py:41` | Low | Improve markdown fence pattern |
| Duplicated preprocessing | `sft_oracle_llama70b_qlora.py` | Medium | Extract to shared module |

---

## Medium-Priority Enhancements

| Enhancement | File | Impact |
|-------------|------|--------|
| Store model hashes in manifest | `package_oracle_model.py` | Better provenance tracking |
| Add eval dataset support | `sft_oracle_llama70b_qlora.py` | Feature parity with LoRA |
| Named tuple for hook returns | `sql_guardrail.py` | Better type safety |
| Parameterized model path | `inference/vllm_config.yaml` | Production-ready |
| Device map as parameter | `train_util_4bit.py` | Flexibility for distributed training |

---

## Low-Priority Improvements

- Add comprehensive unit tests
- Add integration tests for full pipeline
- Add CLI wrappers for common operations
- Add model card generation
- Add experiment tracking (MLflow, Weights & Biases)

---

## Summary

**Overall Quality:** ⭐⭐⭐⭐ (4/5)

The codebase is well-structured, documented, and production-ready with minor refinements needed. All core functionality is present and follows best practices. The main issues are:

1. Python 3.8 compatibility for type hints
2. Missing path validations
3. Code duplication between LoRA and QLoRA scripts

These are straightforward to fix and don't affect functionality.
