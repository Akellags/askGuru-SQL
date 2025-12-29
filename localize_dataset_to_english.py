"""
Localize Oracle SFT dataset from Chinese to English.
Converts all Chinese text markers and labels to English equivalents.
"""
import json
import os
from typing import Any, Dict

# Mapping of Chinese to English replacements
CHINESE_TO_ENGLISH_MAP = {
    "【用户问题】": "[User Question]",
}


def localize_text(text: str) -> str:
    """Replace Chinese text with English equivalents."""
    for chinese, english in CHINESE_TO_ENGLISH_MAP.items():
        text = text.replace(chinese, english)
    return text


def localize_example(example: Dict[str, Any]) -> Dict[str, Any]:
    """Localize a single training example."""
    if "conversations" in example:
        for conv in example["conversations"]:
            if "content" in conv:
                conv["content"] = localize_text(conv["content"])
    return example


def localize_json_file(input_path: str, output_path: str) -> int:
    """
    Localize a JSON file (JSONL format or standard JSON array).
    Returns count of modified examples.
    """
    count = 0
    
    # Try JSONL format first
    if input_path.endswith(".jsonl"):
        with open(input_path, "r", encoding="utf-8") as f_in:
            with open(output_path, "w", encoding="utf-8") as f_out:
                for line in f_in:
                    if line.strip():
                        example = json.loads(line)
                        example = localize_example(example)
                        f_out.write(json.dumps(example, ensure_ascii=False) + "\n")
                        count += 1
    else:
        # Standard JSON array format
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if isinstance(data, list):
            data = [localize_example(example) for example in data]
            count = len(data)
        elif isinstance(data, dict):
            # Handle single dict or dict with nested structure
            data = localize_example(data)
            count = 1
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    return count


def main():
    """Localize all dataset files."""
    base_dir = "data/oracle_sft_conversations"
    
    files_to_localize = [
        "oracle_sft_conversations_train.json",
        "oracle_sft_conversations_val.json",
        "oracle_sft_conversations_test.json",
        "oracle_sft_conversations_full.json",
    ]
    
    print("Localizing dataset to English...\n")
    
    for filename in files_to_localize:
        input_path = os.path.join(base_dir, filename)
        
        if not os.path.exists(input_path):
            print(f"Skipping {filename} - file not found")
            continue
        
        print(f"Processing {filename}...")
        count = localize_json_file(input_path, input_path)
        print(f"Localized {count} examples\n")
    
    print("All files localized successfully!")


if __name__ == "__main__":
    main()
