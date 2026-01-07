import argparse, json, os
from pathlib import Path
from typing import List, Dict, Any
import faiss

# Use the refactored EmbedderManager
from embedder import EmbedderManager

def load_table_cards(tables_dir: Path) -> Dict[str, Dict[str, Any]]:
    cards = {}
    for p in sorted(Path(tables_dir).glob("*.json")):
        obj = json.loads(p.read_text(encoding="utf-8"))
        t = str(obj.get("table", obj.get("table_name", ""))).upper().strip()
        if t: cards[t] = obj
    return cards

def adapt_table_cards(cards: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    adapted = []
    for name, card in cards.items():
        adapted.append({
            "table_name": name,
            "schema": f"CREATE TABLE {name} (...)",
            "description": card.get("purpose") or card.get("notes") or card.get("description", ""),
            "columns": card.get("columns", []),
            "essential_columns": card.get("essential_columns", []),
            "relationships": " | ".join([f"{fk['column']} -> {fk['ref_table']}.{fk['ref_column']}" 
                                        for fk in card.get("fks", [])]),
            "common_queries": " | ".join(card.get("join_hints", []))
        })
    return adapted

def adapt_fewshots(fewshots_path: Path) -> List[Dict[str, Any]]:
    adapted = []
    if not fewshots_path.exists(): return adapted
    for line in fewshots_path.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        ex = json.loads(line)
        adapted.append({
            "question": ex.get("intent", ""),
            "sql": ex.get("sql", ""),
            "db_info": " ".join(ex.get("tables", [])),
        })
    return adapted

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables-dir", required=True)
    ap.add_argument("--fewshots", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="BAAI/bge-en-icl")
    ap.add_argument("--hf-home", help="HF_HOME / Cache directory for transformer models")
    args = ap.parse_args()

    # Set environment variables for HF to ensure no permission issues on root
    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home
        os.environ["TRANSFORMERS_CACHE"] = args.hf_home
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = args.hf_home

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    embedder = EmbedderManager(
        model_name=args.model,
        index_dir=str(out_dir),
        cache_dir=args.hf_home
    )
    
    # Tables
    cards = load_table_cards(Path(args.tables_dir))
    adapted_cards = adapt_table_cards(cards)
    table_embs, table_ids = embedder.create_table_embeddings(adapted_cards)
    table_index = embedder.build_faiss_index(table_embs)
    embedder.save_index(table_index, table_ids, "tables")
    
    # Fewshots
    adapted_fs = adapt_fewshots(Path(args.fewshots))
    if adapted_fs:
        fs_embs, fs_ids = embedder.create_fewshot_embeddings(adapted_fs)
        fs_index = embedder.build_faiss_index(fs_embs)
        embedder.save_index(fs_index, fs_ids, "fewshots")

    print(f"DONE: Dense indices written to {out_dir}")

if __name__ == "__main__":
    main()
