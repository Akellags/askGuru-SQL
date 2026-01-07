import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

from .embedder import EmbedderManager
from .planner import RAGPlanner, BM25Index, merge_rrf
from .evidence_composer import EvidenceComposer

class RAGEngine:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.index_dir = Path(config["index_dir"])
        self.vectors_dir = Path(config["vectors_dir"])
        self.tables_dir = Path(config["tables_dir"])
        self.fewshots_path = Path(config["fewshots"])
        
        # Initialize Embedder
        self.embedder = EmbedderManager(
            model_name=config.get("embedder", {}).get("model_name", "BAAI/bge-en-icl"),
            index_dir=str(self.vectors_dir),
            cache_dir=config.get("embedder", {}).get("cache_dir")
        )
        
        # Load Table Cards
        self.cards = self._load_cards()
        
        # Initialize Planner
        joins_path = Path(config.get("joins", ""))
        ext_joins = []
        if joins_path.exists():
            data = json.loads(joins_path.read_text(encoding="utf-8"))
            # Simple list load for external joins
            if isinstance(data, list): ext_joins = data
        
        self.planner = RAGPlanner(self.cards, external_joins=ext_joins)
        
        # Initialize Evidence Composer
        self.composer = EvidenceComposer(Path(config.get("evidence_policies", "src/api/rag/data/evidence_policies.json")))
        
        # Load Indices
        self.tables_bm25 = BM25Index.load(self.index_dir / "tables_bm25.json")
        self.fewshots_bm25 = BM25Index.load(self.index_dir / "fewshots_bm25.json")
        
        # Load Vector Indices
        self.table_index, self.table_ids = self.embedder.load_index("tables")
        self.fewshot_index, self.fewshot_ids = self.embedder.load_index("fewshots")

    def _load_cards(self) -> Dict[str, Dict]:
        out = {}
        for p in sorted(self.tables_dir.glob("*.json")):
            obj = json.loads(p.read_text(encoding="utf-8"))
            t = str(obj.get("table", "")).upper()
            if t: out[t] = obj
        return out

    def get_dynamic_context(self, question: str, top_tables: int = 8, top_fewshots: int = 4) -> Dict[str, Any]:
        # 1. Hybrid Table Search
        bm25_hits = [h["doc_id"] for h in self.tables_bm25.search(question, topk=50)]
        
        q_vec = self.embedder.embed_query(question)
        D, I = self.table_index.search(np.array([q_vec]).astype("float32"), 50)
        dense_hits = [self.table_ids[i] for i in I[0] if i != -1]
        
        merged_tables = [t for t, _ in merge_rrf(bm25_hits, dense_hits)[:top_tables]]
        
        # 2. Plan (Connect Tables & Select Columns)
        connected = self.planner.connect_tables(merged_tables, max_tables=top_tables)
        cols = self.planner.pick_columns(connected)
        join_conds = self.planner.get_join_conds(connected)
        
        # 3. Hybrid Few-shot Search
        fs_bm25 = [h["doc_id"] for h in self.fewshots_bm25.search(question, topk=30)]
        D_fs, I_fs = self.fewshot_index.search(np.array([q_vec]).astype("float32"), 30)
        fs_dense = [self.fewshot_ids[i] for i in I_fs[0] if i != -1]
        
        merged_fs = merge_rrf(fs_bm25, fs_dense)
        
        # Load fewshot details
        exmap = {}
        if self.fewshots_path.exists():
            for line in self.fewshots_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    ex = json.loads(line)
                    exmap[f"example_{ex.get('id')}"] = ex # Match EmbedderManager ID format
        
        picked_fewshots = []
        for doc_id, _ in merged_fs:
            ex = exmap.get(doc_id)
            if ex:
                picked_fewshots.append(ex)
                if len(picked_fewshots) >= top_fewshots: break
        
        # 4. Compose Evidence
        plan_results = {"join_conds": join_conds}
        evidence = self.composer.compose(question, plan_results)
        
        return {
            "tables": connected,
            "columns": cols,
            "join_hints": join_conds,
            "fewshots": picked_fewshots,
            "evidence": evidence
        }
