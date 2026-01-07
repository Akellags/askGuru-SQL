import json, argparse, re, math, os
from pathlib import Path
from typing import Dict, List, Any

# Internal import from the refactored package
# Note: When running as a script, we use the local BM25 logic to keep it standalone for CLI
def _tok(s: str) -> List[str]:
    return [t.lower() for t in re.findall(r"[A-Za-z0-9_]+", s or "")]

def _bm25_build(docs: Dict[str, List[str]], k1=1.5, b=0.75) -> Dict[str, Any]:
    N = len(docs) or 1
    doc_len = {doc_id: len(tokens) for doc_id, tokens in docs.items()}
    avgdl = sum(doc_len.values()) / float(N)
    df = {}
    for doc_id, tokens in docs.items():
        seen = set(tokens)
        for t in seen:
            df[t] = df.get(t, 0) + 1
    idf = {}
    for t, d in df.items():
        idf[t] = math.log((N - d + 0.5) / (d + 0.5) + 1.0)
    postings = {}
    for doc_id, tokens in docs.items():
        tf = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        for t, f in tf.items():
            postings.setdefault(t, {})[doc_id] = f
    return {
        "k1": k1, "b": b, "avgdl": avgdl,
        "doc_len": doc_len, "idf": idf, "postings": postings,
    }

def load_table_cards(tables_dir: Path) -> Dict[str, Dict[str, Any]]:
    cards = {}
    for p in sorted(Path(tables_dir).glob("*.json")):
        obj = json.loads(p.read_text(encoding="utf-8"))
        t = str(obj.get("table", obj.get("table_name", ""))).upper().strip()
        if t: cards[t] = obj
    return cards

def table_docs(cards: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
    docs = {}
    for t, c in cards.items():
        pieces = [t, c.get("notes", "") or c.get("description", "") or ""]
        for col in c.get("columns", []) or []:
            pieces.append(str(col.get("name", "")))
            pieces.append(str(col.get("desc", col.get("description", ""))))
        for fk in c.get("fks", []) or []:
            pieces.append(str(fk.get("ref_table", "")))
        for j in c.get("join_hints", []) or []:
            pieces.append(str(j))
        docs[t] = _tok(" ".join(pieces))
    return docs

def fewshot_docs(fewshots_path: Path) -> Dict[str, List[str]]:
    docs = {}
    if not fewshots_path.exists(): return docs
    for line in fewshots_path.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        ex = json.loads(line)
        ex_id = ex.get("id") or ex.get("idx")
        if ex_id is not None:
            doc_id = f"EX{ex_id}"
            pieces = [str(ex.get("intent", "")), " ".join(ex.get("tables", [])), str(ex.get("sql", ""))]
            docs[doc_id] = _tok(" ".join(pieces))
    return docs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables-dir", required=True)
    ap.add_argument("--fewshots", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--hf-home", help="Ignored (for consistency with build_vectors.py)")
    ap.add_argument("--hf-token", help="Ignored (for consistency with build_vectors.py)")
    args = ap.parse_args()

    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home
        os.environ["TRANSFORMERS_CACHE"] = args.hf_home

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cards = load_table_cards(Path(args.tables_dir))
    
    tbl_idx = _bm25_build(table_docs(cards))
    (out_dir / "tables_bm25.json").write_text(json.dumps(tbl_idx), encoding="utf-8")
    
    fs_docs = fewshot_docs(Path(args.fewshots))
    fs_idx = _bm25_build(fs_docs)
    (out_dir / "fewshots_bm25.json").write_text(json.dumps(fs_idx), encoding="utf-8")

    print(f"DONE: BM25 indices built at {out_dir}")

if __name__ == "__main__":
    main()
