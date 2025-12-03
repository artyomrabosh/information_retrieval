from __future__ import annotations

from typing import List, Tuple, Set
from core.tokenizer import Tokenizer
import math

SERVICE_TOKENS = set(["OR", "AND", "NEAR", "ADJ", "NOT"])

class RankerL1:
    """Простое линейное L1-ранжирование на базе текстовых фич.

    Модель: score(q, d) = w · f(q, d), где f — вектор признаков:
      * overlap — число уникальных термов запроса, встречающихся в документе
      * tf_sum — суммарная частота термов запроса в документе
      * proximity — мера близости термов запроса (на основе расстояний между словами)
      * len_norm — нормированная длина документа

    Веса можно в дальнейшем обучать по qrel'ам, минимизируя L1-ошибку (|y - w·f|),
    но здесь задана простая инициализация по умолчанию.
    """

    def __init__(self, inverted_index, direct_index):
        self.inverted_index = inverted_index
        self.direct_index = direct_index
        self.tokenizer = Tokenizer()

        self.w_overlap = 1.0
        self.w_tf_sum = 0.2
        self.w_proximity = 2.0
        self.w_len_norm = 0.1

        self.features = {"overlap": self._feature_overlap, 
                         "tf_sum": self._feature_tf_idf_sum, 
                         "prox": self._feature_proximity,
                         "len_norm": self._feature_log_doc_len}
        self.weights = [1 for _ in self.features] + [1]

    def rank(self, query: str, candidates: Set[str], top_k: int = 10) -> List[Tuple[str, float]]:
        """Отранжировать документы по запросу.

        1) извлекаем термы запроса
        2) получаем множество кандидатов по инвертированному индексу
        3) считаем признаки и скор для каждого кандидата
        4) возвращаем top_k пар (doc_id, score)
        """
        query_terms = self._extract_query_terms(query)
        if not query_terms:
            return []

        scored: List[Tuple[str, float]] = []

        for doc_id in candidates:
            score = self._score(query_terms, doc_id)
            scored.append((doc_id, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        if top_k is not None and top_k > 0:
            scored = scored[:top_k]
        return scored
    
    def get_features(self, query: str, candidates: Set[str]) -> List[Tuple[str, float]]:
        query_terms = self._extract_query_terms(query)
        if not query_terms:
            return []

        scored: List[Tuple[str, float]] = []

        for doc_id in candidates:
            features = self._get_features(query_terms, doc_id)
            scored.append((doc_id, features))
        return scored

    def _extract_query_terms(self, query: str) -> List[str]:
        """Простая токенизация запроса (без булевой логики)."""
        tokens = self.tokenizer.tokenize(query)
        return [t for t, _ in tokens if t not in SERVICE_TOKENS]

    def _compute_idf_cache(self):
        """Кэширование IDF значений для всех термов в индексе."""
        self.idf_cache = {}
        N = len(self.inverted_index.documents)  # Общее количество документов
        
        for term in self.inverted_index.doc_index:
            df = len(self.inverted_index._get_numeric_doc_ids(term))
            self.idf_cache[term] = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
        
        print(self.idf_cache)

    def _feature_overlap(self, query_terms: List[str], doc_id: str) -> float:
        terms_in_doc = self.direct_index.get_terms(doc_id)
        if not terms_in_doc:
            return 0.0
        uniq_q = set(query_terms)
        return float(len(uniq_q.intersection(terms_in_doc.keys())))

    def _feature_tf_idf_sum(self, query_terms: List[str], doc_id: str) -> float:
        terms_in_doc = self.direct_index.get_terms(doc_id)
        if not terms_in_doc:
            return 0.0
        tf_sum = 0
        for term in set(query_terms):
            positions = self.direct_index.get_positions(doc_id, term)
            tf_sum += len(positions) * self.idf_cache.get(term, 0)
        return float(tf_sum)

    def _feature_proximity(self, query_terms: List[str], doc_id: str) -> float:
        """Фича близости термов: чем ближе термы, тем больше значение (0..1]."""
        dists: List[int] = []

        for i in range(len(query_terms) - 1):
            t1 = query_terms[i]
            t2 = query_terms[i + 1]

            pos1 = self.direct_index.get_positions(doc_id, t1)
            pos2 = self.direct_index.get_positions(doc_id, t2)

            if not pos1 or not pos2:
                continue

            # минимальное расстояние между вхождениями t1 и t2 (два отсортированных списка)
            min_dist = None
            j = 0
            k = 0
            while j < len(pos1) and k < len(pos2):
                d = abs(pos1[j] - pos2[k])
                if min_dist is None or d < min_dist:
                    min_dist = d
                if pos1[j] < pos2[k]:
                    j += 1
                else:
                    k += 1

            if min_dist is not None and min_dist > 0:
                dists.append(min_dist)

        if not dists:
            return 0.0

        avg_dist = sum(dists) / len(dists)
        # преобразуем расстояние в «близость» (меньшее расстояние -> больше значение)
        return 1.0 / (1.0 + avg_dist)
    
    def _feature_log_doc_len(self, query_terms: List[str], doc_id: str) -> float:
        doc_len = self.direct_index.get_document_length(doc_id)
        if doc_len == 0:
            return 0.0
        # нормируем длину документа через лог, чтобы не заваливать длинные документы
        return math.log(1.0 + doc_len) / (len(query_terms) + 1e-6)

    def _score(self, query_terms: List[str], doc_id: str) -> float:
        """Линейная модель: score = w · f."""
        features = []
        for feature_name, feature in self.features.items():
            features.append(feature(query_terms, doc_id))
        
        features.append(1) # bias в конце
        return sum(w_i * x_i for w_i, x_i in zip(self.weights, features))
    
    def _get_features(self, query_terms: List[str], doc_id: str):
        features = []
        for feature_name, feature in self.features.items():
            features.append(feature(query_terms, doc_id))
        return features