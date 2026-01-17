from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import os
import numpy as np

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

class VectorIndex:
    """Простейший векторный поиск по dot product.

    Варианты использования:
      * full-scan по всей матрице эмбеддингов (для демо/небольших коллекций)
      * поиск/скоры только по подмножеству doc_ids (ранкпул)
    """

    def __init__(self, index_dir: str = "./datasets/vector_search_data", normalize: bool = False):
        self.normalize = normalize
        print("Loading FAISS index...")
        self.index = faiss.read_index(f"{index_dir}/faiss_index.bin")
        print("Loading embedding model...")
        self.model = self._load_or_download_model()
        self.passages_df = pd.read_parquet(f"{index_dir}/passages.parquet")
        
        self.faiss_to_local_mapping: List[str] = list(self.passages_df.doc_id)

    def _load_or_download_model(self):
        """Check for model locally, download if not found, and save it"""
        
        self.model_name='sentence-transformers/all-MiniLM-L6-v2'
        self.local_path='./models/sentence-transformer'

        model_exists = os.path.exists(self.local_path) and any(
            fname.endswith('.json') or fname.endswith('.bin') or fname.endswith('.pt')
            for fname in os.listdir(self.local_path)
        )
        
        if model_exists:
            print(f"Loading model from local path: {self.local_path}")
            try:
                return SentenceTransformer(self.local_path)
            except Exception as e:
                print(f"Error loading local model, downloading fresh: {e}")
                model_exists = False
        
        if not model_exists:
            print(f"Downloading model: {self.model_name}")
            model = SentenceTransformer(self.model_name)
            
            os.makedirs(self.local_path, exist_ok=True)
            
            print(f"Saving model to: {self.local_path}")
            model.save(self.local_path)
            return
    
    def encode_query(self, query_text):
        """Кодирует запрос в эмбеддинг"""
        embeddings = self.model.encode(query_text)
        return embeddings

    def search(
        self,
        query_text: str,
        top_k: int = 1000,
    ) -> List[Tuple[str, float]]:
        """Возвращает top_k (doc_id, score) по dot product."""

        """
        Поиск по запросу
        
        Args:
            query_text: текст запроса
            top_k: сколько результатов вернуть
            return_passages: возвращать ли пассажи или только документы
        
        Returns:
            Список результатов
        """
        query_embedding = self.encode_query(query_text)
        
        distances, indices = self.index.search(query_embedding.reshape(1, 384), top_k)
        doc_ids = [self.faiss_to_local_mapping[i] for i in indices[0].tolist()]
        return doc_ids, distances
