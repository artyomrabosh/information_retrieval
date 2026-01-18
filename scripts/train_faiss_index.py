import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

def create_ivf_index(embeddings_path: str = "../datasets/vector_search_data/embeddings.npy",
                     output_path: str = "../datasets/vector_search_data/faiss_index_ivf.bin",
                     nlist: int = 100,  # количество кластеров (центроидов)
                     nprobe: int = 10): # сколько кластеров просматривать при поиске
    """
    Создает IVF индекс для ускоренного поиска.
    
    Args:
        embeddings_path: путь к файлу с эмбеддингами
        output_path: куда сохранить индекс
        nlist: количество кластеров (чем больше, тем точнее, но медленнее)
        nprobe: сколько кластеров просматривать при поиске (чем больше, тем точнее)
    """
    
    print("Loading embeddings...")
    embeddings = np.load(embeddings_path).astype('float32')
    dimension = embeddings.shape[1]
    
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Dimension: {dimension}")
    
    quantizer = faiss.IndexFlatIP(dimension)
    
    index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
    
    print(f"Training IVF index with nlist={nlist}...")
    
    index.train(embeddings)
    
    print(f"Adding {len(embeddings)} vectors to IVF index...")
    
    index.add(embeddings)
    
    index.nprobe = nprobe
    
    faiss.write_index(index, output_path)
    
    print(f"✅ IVF index created and saved to {output_path}")
    print(f"   nlist: {nlist}, nprobe: {nprobe}")
    print(f"   Total vectors: {index.ntotal}")
    
    return index

def create_pq_index(embeddings_path: str = "../datasets/vector_search_data/embeddings.npy",
                    output_path: str = "../datasets/vector_search_data/faiss_index_pq.bin",
                    m: int = 16,  # количество сегментов (должно делиться на 384)
                    bits: int = 8, # бит на сегмент (8 = 256 центроидов на сегмент)
                    nlist: int = 100,  # для IVF часть
                    nprobe: int = 10):
    """
    Создает IVF+PQ индекс для экономии памяти и ускоренного поиска.
    Сочетает IVF (кластеризацию) и PQ (сжатие).
    
    Args:
        m: количество сегментов (384 / m должно быть целым)
        bits: бит на сегмент (обычно 8)
    """
    
    print("Loading embeddings...")
    embeddings = np.load(embeddings_path).astype('float32')
    dimension = embeddings.shape[1]
    
    # Проверяем, что dimension делится на m
    if dimension % m != 0:
        print(f"Warning: dimension {dimension} is not divisible by m={m}")
        print(f"Using m={dimension//16} instead")
        m = dimension // 16
    
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Dimension: {dimension}, m={m}, bits={bits}")
    
    # 1. Создаем квантователь для IVF
    quantizer = faiss.IndexFlatIP(dimension)
    
    # 2. Создаем IVF+PQ индекс
    index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, bits, faiss.METRIC_INNER_PRODUCT)
    
    print(f"Training IVF+PQ index...")
    
    # 3. Обучаем индекс
    index.train(embeddings)
    
    print(f"Adding {len(embeddings)} vectors...")
    
    # 4. Добавляем векторы
    index.add(embeddings)
    
    # 5. Устанавливаем nprobe
    index.nprobe = nprobe
    
    faiss.write_index(index, output_path)
    
    original_size = embeddings.nbytes / 1024**3  # в ГБ
    pq_size = (index.ntotal * m * bits) / (8 * 1024**3)  # в ГБ
    
    print(f"✅ IVF+PQ index created and saved to {output_path}")
    print(f"   nlist: {nlist}, nprobe: {nprobe}, m={m}, bits={bits}")
    print(f"   Total vectors: {index.ntotal}")
    print(f"   Compression: {original_size:.2f} GB -> {pq_size:.2f} GB ({original_size/pq_size:.1f}x)")
    
    return index

def create_ivf_pq_flat_index(embeddings_path: str = "../datasets/vector_search_data/embeddings.npy",
                            output_path: str = "../datasets/vector_search_data/faiss_index_ivf_pq_flat.bin",
                            nlist: int = 100,
                            nprobe: int = 10):
    """
    Создает плоский (flat) IVF индекс для точного поиска.
    В отличие от PQ, хранит полные векторы в кластерах.
    """
    
    print("Loading embeddings...")
    embeddings = np.load(embeddings_path).astype('float32')
    dimension = embeddings.shape[1]
    
    print(f"Embeddings shape: {embeddings.shape}")
    
    quantizer = faiss.IndexFlatIP(dimension)
    index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
    
    print(f"Training IVFFlat index...")
    index.train(embeddings)
    
    print(f"Adding vectors...")
    index.add(embeddings)
    
    index.nprobe = nprobe
    
    faiss.write_index(index, output_path)
    
    print(f"✅ IVFFlat index created and saved to {output_path}")
    print(f"   nlist: {nlist}, nprobe: {nprobe}")
    print(f"   Total vectors: {index.ntotal}")
    
    return index

def test_index_speed(query_embedding, index, n_queries=1000):
    """
    Тестирует скорость поиска в индексе.
    """
    import time
    
    query_embedding = query_embedding.astype('float32')
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)
    
    for _ in range(10):
        _ = index.search(query_embedding, 10)
    
    start = time.time()
    for _ in range(n_queries):
        _ = index.search(query_embedding, 10)
    end = time.time()
    
    avg_time = (end - start) * 1000 / n_queries  # в мс
    print(f"Average search time: {avg_time:.2f} ms")
    
    return avg_time

class OptimizedVectorIndex:
    def __init__(self, 
                 index_dir: str = "./datasets/vector_search_data",
                 index_type: str = "ivf",  # "flat", "ivf", "ivf_pq", "ivf_pq_flat"
                 normalize: bool = False,
                 nprobe: int = 10):
        
        self.normalize = normalize
        self.index_type = index_type
        self.nprobe = nprobe
        
        print(f"Loading {index_type} FAISS index...")
        
        if index_type == "ivf":
            index_path = f"{index_dir}/faiss_index_ivf.bin"
        elif index_type == "ivf_pq":
            index_path = f"{index_dir}/faiss_index_pq.bin"
        elif index_type == "ivf_pq_flat":
            index_path = f"{index_dir}/faiss_index_ivf_pq_flat.bin"
        else:
            index_path = f"{index_dir}/faiss_index.bin"  # плоский индекс
        
        self.index = faiss.read_index(index_path)
        
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = nprobe
            print(f"Set nprobe to {nprobe}")
        
        print("Loading embedding model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        print("Loading passages metadata...")
        self.passages_df = pd.read_parquet(f"{index_dir}/passages.parquet")
        
        print("Loading embeddings for MaxP calculations...")
        self.embeddings = np.load(f"{index_dir}/embeddings.npy")
        
        self._build_mappings()
        
        self.query_cache = {}
        self.maxp_cache = {}
        
        print(f"✅ {index_type.upper()} index ready")
        print(f"   Total passages: {self.index.ntotal}")
        print(f"   Total documents: {len(self.doc_to_passage_indices)}")
    
    def _build_mappings(self):
        """Строит маппинги doc_id -> passage_indices."""
        self.doc_to_passage_indices = {}
        self.passage_idx_to_doc = []
        
        for idx, (_, row) in enumerate(self.passages_df.iterrows()):
            doc_id = row['doc_id']
            
            if doc_id not in self.doc_to_passage_indices:
                self.doc_to_passage_indices[doc_id] = []
            
            self.doc_to_passage_indices[doc_id].append(idx)
            self.passage_idx_to_doc.append(doc_id)
    
    def encode_query(self, query_text: str, use_cache: bool = True) -> np.ndarray:
        """Кодирует запрос в эмбеддинг."""
        if use_cache and query_text in self.query_cache:
            return self.query_cache[query_text]
        
        embedding = self.model.encode(
            query_text, 
            normalize_embeddings=self.normalize,
            show_progress_bar=False
        )
        
        embedding = embedding.reshape(1, -1).astype('float32')
        
        if use_cache:
            self.query_cache[query_text] = embedding
        
        return embedding
    
    def search(self, query_text: str, top_k: int = 1000) -> list:
        """Быстрый поиск с использованием IVF/PQ индекса."""
        query_embedding = self.encode_query(query_text, use_cache=True)
        
        distances, indices = self.index.search(query_embedding, top_k * 10)
        
        doc_scores = {}
        
        for score, passage_idx in zip(distances[0], indices[0]):
            if passage_idx < 0:
                continue
            
            doc_id = self.passage_idx_to_doc[passage_idx]
            
            if doc_id not in doc_scores or score > doc_scores[doc_id]:
                doc_scores[doc_id] = float(score)
        
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        self.maxp_cache[query_text] = dict(sorted_docs)
        
        return sorted_docs
    
    def get_maxP_for_candidates(
        self, 
        query_text: str, 
        candidate_doc_ids: list,
        batch_size: int = 100
    ) -> dict:
        """
        Получает MaxP скоры для кандидатов.
        Проверяет кэш для каждого документа отдельно.
        """
        if query_text not in self.query_cache:
            query_embedding = self.encode_query(query_text, use_cache=True)
        else:
            query_embedding = self.query_cache[query_text]
        
        cached_scores = {}
        to_compute_docs = []
        
        if query_text in self.maxp_cache:
            cached_results = self.maxp_cache[query_text]
            for doc_id in candidate_doc_ids:
                if doc_id in cached_results:
                    cached_scores[doc_id] = cached_results[doc_id]
                else:
                    to_compute_docs.append(doc_id)
        else:
            to_compute_docs = candidate_doc_ids
        
        if not to_compute_docs:
            return cached_scores
        
        computed_scores = self._compute_maxp_batch(
            query_embedding, 
            to_compute_docs, 
            batch_size
        )
        
        if query_text in self.maxp_cache:
            self.maxp_cache[query_text].update(computed_scores)
        else:
            self.maxp_cache[query_text] = computed_scores
        
        result = {**cached_scores, **computed_scores}
        
        for doc_id in candidate_doc_ids:
            if doc_id not in result:
                result[doc_id] = 0.0
        
        return result
    
    def _compute_maxp_batch(self, query_embedding, doc_ids, batch_size):
        """Вычисляет MaxP для батча документов."""
        if not doc_ids:
            return {}
        
        results = {}
        query_flat = query_embedding.flatten()
        
        for i in range(0, len(doc_ids), batch_size):
            batch_doc_ids = doc_ids[i:i + batch_size]
            
            for doc_id in batch_doc_ids:
                if doc_id not in self.doc_to_passage_indices:
                    results[doc_id] = 0.0
                    continue
                
                passage_indices = self.doc_to_passage_indices[doc_id]
                if not passage_indices:
                    results[doc_id] = 0.0
                    continue
                
                passage_embeddings = self.embeddings[passage_indices]
                
                dots = np.dot(passage_embeddings, query_flat)
                
                max_score = float(np.max(dots))
                results[doc_id] = max_score
        
        return results

def main():
    embeddings_path = "./datasets/vector_search_data/embeddings.npy"
    index_dir = "./datasets/vector_search_data"
    
    print("=" * 60)
    print("Creating IVF index...")
    print("=" * 60)
    
    ivf_index = create_ivf_index(
        embeddings_path=embeddings_path,
        output_path=f"{index_dir}/faiss_index_ivf.bin",
        nlist=100,    # 100 кластеров
        nprobe=10     # просматривать 10 кластеров при поиске
    )
    
    print("\n" + "=" * 60)
    print("Creating IVF+PQ index...")
    print("=" * 60)
    
    pq_index = create_pq_index(
        embeddings_path=embeddings_path,
        output_path=f"{index_dir}/faiss_index_pq.bin",
        m=16,         # 16 сегментов (384/16=24)
        bits=8,       # 8 бит на сегмент
        nlist=100,
        nprobe=10
    )
    
    print("\n" + "=" * 60)
    print("Testing different index types...")
    print("=" * 60)
    
    test_query = "машинное обучение"
    model = SentenceTransformer('all-MiniLM-L6-v2')
    test_embedding = model.encode(test_query, normalize_embeddings=False).reshape(1, -1)
    
    print("\nTesting search speed:")
    
    print("IVF index:")
    test_index_speed(test_embedding, ivf_index, n_queries=100)
    
    print("\nIVF+PQ index:")
    test_index_speed(test_embedding, pq_index, n_queries=100)
    
    print("\nCreating flat index for comparison...")
    flat_index = faiss.IndexFlatIP(384)
    embeddings = np.load(embeddings_path).astype('float32')
    flat_index.add(embeddings)
    
    print("Flat index:")
    test_index_speed(test_embedding, flat_index, n_queries=100)
    

if __name__ == "__main__":
    main()