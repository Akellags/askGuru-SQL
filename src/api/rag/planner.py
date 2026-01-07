import json, re
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict, deque
import numpy as np

# Helper for BM25
_WORD_RE = re.compile(r"[A-Za-z0-9_]+")
def _tok(s: str): return [t.lower() for t in _WORD_RE.findall(s or "")]

class BM25Index:
    def __init__(self, obj: Dict[str, Any]):
        self.k1 = obj.get("k1", 1.5)
        self.b = obj.get("b", 0.75)
        self.avgdl = obj.get("avgdl", 0.0)
        self.doc_len = obj.get("doc_len", {})
        self.idf = obj.get("idf", {})
        self.postings = obj.get("postings", {})

    @classmethod
    def load(cls, path: Path):
        if not path.exists():
            return None
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def search(self, query: str, topk: int = 10):
        scores = defaultdict(float)
        for q in _tok(query):
            if q not in self.postings: continue
            idf_q = self.idf.get(q, 0.0)
            for doc_id, f in self.postings[q].items():
                dl = self.doc_len.get(doc_id, 0)
                denom = f + self.k1 * (1 - self.b + self.b * (dl / (self.avgdl or 1e-9)))
                scores[doc_id] += idf_q * (f * (self.k1 + 1)) / (denom or 1e-9)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:topk]
        return [{"doc_id": d, "score": float(s)} for d, s in ranked]

def merge_rrf(list1: List[str], list2: List[str], k: int = 60) -> List[tuple]:
    """Reciprocal Rank Fusion"""
    scores = defaultdict(float)
    for i, doc_id in enumerate(list1): scores[doc_id] += 1.0 / (k + i + 1)
    for i, doc_id in enumerate(list2): scores[doc_id] += 1.0 / (k + i + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

class RAGPlanner:
    def __init__(self, cards: Dict[str, Dict], external_joins: List[str] = None):
        self.cards = cards
        self.graph = self._build_graph(external_joins)

    def _build_graph(self, external_joins):
        g = defaultdict(lambda: {"neighbors": {}})
        def add(a, b, cond):
            a, b = a.upper(), b.upper()
            g[a]["neighbors"].setdefault(b, {"conds": []})
            g[b]["neighbors"].setdefault(a, {"conds": []})
            if cond and cond not in g[a]["neighbors"][b]["conds"]: g[a]["neighbors"][b]["conds"].append(cond)
            if cond and cond not in g[b]["neighbors"][a]["conds"]: g[b]["neighbors"][a]["conds"].append(cond)
        
        for t, c in self.cards.items():
            for fk in c.get("fks", []):
                ref = str(fk.get("ref_table", "")).upper()
                col = str(fk.get("column", "")).upper()
                refc = str(fk.get("ref_column", "")).upper()
                if ref: add(t, ref, f"{t}.{col} = {ref}.{refc}")
            for j in c.get("join_hints", []):
                m = re.match(r'([A-Z0-9_]+\.[A-Z0-9_]+)\s*=\s*([A-Z0-9_]+\.[A-Z0-9_]+)', str(j).upper())
                if m: add(m.group(1).split(".")[0], m.group(2).split(".")[0], m.group(0))
        
        for cond in (external_joins or []):
            m = re.match(r'([A-Z0-9_]+\.[A-Z0-9_]+)\s*=\s*([A-Z0-9_]+\.[A-Z0-9_]+)', str(cond).upper())
            if m: add(m.group(1).split(".")[0], m.group(2).split(".")[0], m.group(0))
        return g

    def shortest_path(self, src, dst):
        src, dst = src.upper(), dst.upper()
        if src == dst: return [src]
        dq, prev = deque([src]), {src: None}
        while dq:
            u = dq.popleft()
            for v in self.graph[u]["neighbors"]:
                if v not in prev:
                    prev[v] = u; dq.append(v)
                    if v == dst:
                        path = []; cur = v
                        while cur: path.append(cur); cur = prev[cur]
                        return path[::-1]
        return []

    def connect_tables(self, selected: List[str], max_tables: int = 8) -> List[str]:
        if not selected: return []
        out = [selected[0]]
        for t in selected[1:]:
            if t in out: continue
            p = self.shortest_path(out[0], t)
            for node in p:
                if node not in out: out.append(node)
                if len(out) >= max_tables: return out
        return out

    def pick_columns(self, tables: List[str], per_table: int = 16) -> Dict[str, List[str]]:
        out = {}
        for t in tables:
            c = self.cards.get(t, {})
            cols = c.get("columns", [])
            chosen = []
            # Priority 1: Essential
            for ec in c.get("essential_columns", []):
                n = str(ec.get("name", "")).upper()
                if n and n not in chosen: chosen.append(n)
            # Priority 2: Primary Keys
            for pk in c.get("pk", []):
                n = str(pk).upper()
                if n and n not in chosen: chosen.append(n)
            # Priority 3: Rest
            for col in cols:
                n = str(col.get("name", "")).upper()
                if n and n not in chosen: chosen.append(n)
                if len(chosen) >= per_table: break
            out[t] = chosen
        return out

    def get_join_conds(self, path: List[str]) -> List[str]:
        conds = []
        for i in range(len(path) - 1):
            a, b = path[i], path[i+1]
            edge = self.graph[a]["neighbors"].get(b)
            if edge and edge["conds"]: conds.append(edge["conds"][0])
        return list(set(conds))
