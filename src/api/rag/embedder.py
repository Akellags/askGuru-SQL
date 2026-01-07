import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
from pathlib import Path

class EmbedderManager:
    """Handles embedding creation and vector storage for RAG system"""
    
    def __init__(self, 
                 model_name: str = "BAAI/bge-en-icl",
                 index_dir: str = None,
                 cache_dir: str = None,
                 token: str = None):
        self.model_name = model_name
        self.index_dir = Path(index_dir) if index_dir else Path.cwd() / "rag" / "index" / "vectors"
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = cache_dir
        
        print(f"Loading BGE embedder: {model_name}")
        # Use CPU for embedding to save VRAM for the large LLM
        device = "cpu"
        
        # Try loading locally first to avoid 401/expired token issues
        try:
            # We explicitly pass token=False for public models if we don't have a specific token
            # to prevent HF from using an expired token from the environment.
            hf_token = token if token else False
            
            self.model = SentenceTransformer(
                model_name, 
                cache_folder=cache_dir, 
                device=device,
                token=hf_token,
                model_kwargs={"local_files_only": True}
            )
            print("Successfully loaded model from local cache.")
        except Exception as e:
            print(f"Local load failed or model not found in {cache_dir}. Attempting download...")
            self.model = SentenceTransformer(
                model_name, 
                cache_folder=cache_dir, 
                device=device,
                token=token if token else False
            )
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Embedding dimension: {self.embedding_dim}")
    
    def create_table_embeddings(self, table_cards: List[Dict]) -> Tuple[np.ndarray, List[str]]:
        """Create embeddings for table cards"""
        texts = []
        table_ids = []
        
        for card in table_cards:
            text_parts = []
            if 'table_name' in card: text_parts.append(f"Database Table: {card['table_name']}")
            if 'schema' in card: text_parts.append(f"CREATE TABLE Definition: {card['schema']}")
            if 'description' in card: text_parts.append(f"Purpose: {card['description']}")
            
            if 'essential_columns' in card and card['essential_columns']:
                essential_cols = []
                for ec in card['essential_columns']:
                    if isinstance(ec, dict):
                        col_name = ec.get('name', '')
                        col_reason = ec.get('reason', '')
                        if col_name: essential_cols.append(f"{col_name} ({col_reason})" if col_reason else col_name)
                if essential_cols: text_parts.append(f"ESSENTIAL COLUMNS: {'; '.join(essential_cols)}")
            
            if 'columns' in card:
                col_details = []
                for col in card['columns']:
                    if not isinstance(col, dict): continue
                    col_name = col.get('name')
                    if not col_name: continue
                    col_str = f"{col_name} ({col.get('type', 'unknown')})"
                    if col.get('description'): col_str += f" - {col['description']}"
                    col_details.append(col_str)
                if col_details: text_parts.append(f"Columns: {'; '.join(col_details)}")
            
            if 'relationships' in card: text_parts.append(f"Relationships: {card['relationships']}")
            if 'common_queries' in card: text_parts.append(f"Typical queries: {card['common_queries']}")
            
            full_text = " | ".join(text_parts)
            texts.append(full_text)
            table_ids.append(card.get('table_name', f"table_{len(table_ids)}"))
        
        embeddings = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings, table_ids
    
    def create_fewshot_embeddings(self, fewshots: List[Dict]) -> Tuple[np.ndarray, List[str]]:
        """Create embeddings for few-shot examples"""
        texts = []
        example_ids = []
        
        for i, example in enumerate(fewshots):
            text_parts = []
            if 'question' in example: text_parts.append(f"Question: {example['question']}")
            if 'sql' in example:
                sql_clean = example['sql'].strip().replace('\n', ' ')
                text_parts.append(f"SQL Pattern: {sql_clean}")
            if 'db_info' in example: text_parts.append(f"Context: {example['db_info']}")
            
            full_text = " | ".join(text_parts)
            texts.append(full_text)
            example_ids.append(f"example_{i}")
        
        embeddings = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings, example_ids
    
    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Build FAISS index"""
        index = faiss.IndexFlatIP(self.embedding_dim)
        faiss.normalize_L2(embeddings.astype(np.float32))
        index.add(embeddings.astype(np.float32))
        return index
    
    def save_index(self, index: faiss.Index, ids: List[str], index_name: str):
        """Save FAISS index and IDs"""
        index_path = self.index_dir / f"{index_name}.faiss"
        ids_path = self.index_dir / f"{index_name}.ids.json"
        faiss.write_index(index, str(index_path))
        with open(ids_path, 'w') as f:
            json.dump(ids, f, indent=2)
    
    def load_index(self, index_name: str) -> Tuple[faiss.Index, List[str]]:
        """Load FAISS index and IDs"""
        index_path = self.index_dir / f"{index_name}.faiss"
        ids_path = self.index_dir / f"{index_name}.ids.json"
        if not index_path.exists(): raise FileNotFoundError(f"Index not found: {index_path}")
        index = faiss.read_index(str(index_path))
        with open(ids_path, 'r') as f: ids = json.load(f)
        return index, ids
    
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query"""
        embedding = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        return embedding[0]

    def cleanup(self):
        """Free memory"""
        try:
            import torch
            self.model.to('cpu')
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except: pass
