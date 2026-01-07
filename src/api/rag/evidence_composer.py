import re, json
from pathlib import Path
from typing import Dict, Any, List

MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec",
          "January","February","March","April","June","July","August","September","October","November","December"]

class EvidenceComposer:
    def __init__(self, policies_path: Path):
        self.policies = self._load_policies(policies_path)

    def _load_policies(self, path: Path) -> Dict[str, Any]:
        if not path.exists(): return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def extract_hints(self, question: str) -> Dict[str, Any]:
        q = question or ""
        mons = [m for m in MONTHS if re.search(rf'\b{re.escape(m)}[\s\-]?\d{{2,4}}?', q, flags=re.IGNORECASE)]
        years = re.findall(r'\b(20\d{2}|19\d{2})\b', q)
        led = None
        m = re.search(r"\b(?:for|from|in)\s+(ledger|book|set of books)?\s*([A-Za-z][A-Za-z0-9\s\-\(\)]{2,})$", q, flags=re.IGNORECASE)
        if m: led = m.group(2).strip()
        return {"months": sorted(set(mons)), "years": sorted(set(years)), "ledger_hint": led}

    def compose(self, question: str, plan_results: Dict[str, Any]) -> str:
        q = question or ""
        recipe_lines = []

        # Match policy
        for name, spec in self.policies.items():
            try:
                if re.search(spec.get("match_regex",""), q):
                    recipe_lines.append(spec["recipe"])
                    break
            except: continue

        hints = self.extract_hints(q)
        if hints["months"] or hints["years"] or hints["ledger_hint"]:
            parts = []
            if hints["months"]: parts.append("months: " + ",".join(hints["months"]))
            if hints["years"]:  parts.append("years: " + ",".join(hints["years"]))
            if hints["ledger_hint"]: parts.append("ledger hint: " + hints["ledger_hint"])
            recipe_lines.append("Hints: " + "; ".join(parts))

        jh = plan_results.get("join_conds", [])
        if jh: recipe_lines.append("Join hints: " + "; ".join(jh[:5]))

        return "\n".join(recipe_lines).strip()
