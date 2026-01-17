from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Dict

import os
import numpy as np
from collections import defaultdict

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

class VectorIndex:
    """векторный поиск по dot product.

    Варианты использования:
      * full-scan по всей матрице эмбеддингов (для демо/небольших коллекций)
      * поиск/скоры только по подмножеству doc_ids (ранкпул)
    """

    def __init__(self, index_dir: str = "./datasets/vector_search_data", use_memmap: bool = True):
        self.use_memmap = use_memmap
        print("Loading FAISS index...")
        self.index = faiss.read_index(f"{index_dir}/faiss_index.bin")
        print("Loading embedding model...")
        self.model = self._load_or_download_model()
        self.passages_df = pd.read_parquet(f"{index_dir}/passages.parquet")
        
        print("Setting up embeddings storage...")
        self.embeddings = self._setup_embeddings_storage(index_dir)
        
        print("Building mappings...")
        self._build_mappings()

        self.query_cache: Dict[str, np.ndarray] = {}
        self.maxp_cache: Dict[str, Dict[int, float]] = {}

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
    
    def _setup_embeddings_storage(self, index_dir: str) -> np.ndarray:
        """Настраивает хранение эмбеддингов (memmap или загрузка в память)."""
        embeddings_path = f"{index_dir}/embeddings.npy"
        
        if not os.path.exists(embeddings_path):
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
        
        if self.use_memmap:
            print(f"Creating memmap for embeddings from {embeddings_path}")
            
            with open(embeddings_path, 'rb') as f:
                version = np.lib.format.read_magic(f)
                shape, fortran_order, dtype = np.lib.format._read_array_header(f, version)
            
            embeddings = np.memmap(
                embeddings_path,
                dtype=dtype,
                mode='r',  # только чтение
                shape=shape,
            )
            
            print(f"Memmap created: shape={shape}, dtype={dtype}")
            return embeddings
        else:
            print(f"Loading embeddings into memory from {embeddings_path}")
            embeddings = np.load(embeddings_path)
            print(f"Embeddings loaded: shape={embeddings.shape}")
            return embeddings
        
    def _build_mappings(self) -> None:
        """Строит необходимые маппинги для быстрого доступа."""
        self.faiss_to_doc = list(self.passages_df.doc_id)
        self.doc_to_faiss_indices = defaultdict(list)
        
        for faiss_idx, (_, row) in enumerate(self.passages_df.iterrows()):
            doc_id = row['doc_id']
            self.doc_to_faiss_indices[doc_id].append(faiss_idx)
    
    def encode_query(self, query_text, use_cache: bool = True):
        """Кодирует запрос в эмбеддинг"""
        if use_cache and query_text in self.query_cache:
            return self.query_cache[query_text]
        
        embedding = self.model.encode(query_text)
        
        embedding = embedding.reshape(1, -1).astype('float32')
        
        if use_cache:
            self.query_cache[query_text] = embedding
            
        return embedding
    

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
        doc_scores = defaultdict(float)
        
        for score, passage_idx in zip(distances[0], indices[0]):
            if passage_idx < 0:
                continue
            
            doc_id = self.faiss_to_doc[passage_idx]
            if score > doc_scores[doc_id]:
                doc_scores[doc_id] = float(score)
        
        # Сортируем документы
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Сохраняем в кэш
        self.maxp_cache[query_text] = dict(sorted_docs)
        
        return sorted_docs
    
    def get_maxP_for_candidates(
        self, 
        query_text: str, 
        candidate_doc_ids: List[int]
    ) -> Dict[int, float]:
        """
        Получает MaxP скоры для заданных кандидатов по кэшированному запросу.
        
        Args:
            query_text: текст запроса (должен быть в кэше)
            candidate_doc_ids: список ID документов
            
        Returns:
            Словарь {doc_id: maxp_score}
        """
        # Проверяем, есть ли запрос в кэше
        if query_text not in self.query_cache:
            raise ValueError(
                f"Query '{query_text}' not in cache. "
                f"Call search() first or set use_cache=True in encode_query()."
            )
        
        # Проверяем, есть ли результаты в кэше MaxP
        if query_text in self.maxp_cache:
            # Возвращаем только запрошенные кандидаты
            cached_results = self.maxp_cache[query_text]
            return {
                doc_id: cached_results.get(doc_id, 0.0)
                for doc_id in candidate_doc_ids
            }
        
        query_embedding = self.query_cache[query_text]
        
        results = {}
        
        for doc_id in candidate_doc_ids:
            if doc_id not in self.doc_to_faiss_indices:
                results[doc_id] = 0.0
                continue
            
            passage_indices = self.doc_to_faiss_indices[doc_id]
            if not passage_indices:
                results[doc_id] = 0.0
                continue
            
            passage_embeddings = self._get_passage_embeddings(passage_indices)
            query_flat = query_embedding.flatten()
            dots = np.dot(passage_embeddings, query_flat)
            
            max_score = float(np.max(dots))
            results[doc_id] = max_score
        
        return results
    
    def clear_cache(self) -> None:
        """Очищает кэши запросов."""
        self.query_cache.clear()
        self.maxp_cache.clear()
        print("Cache cleared.")
