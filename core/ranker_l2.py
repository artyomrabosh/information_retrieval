import numpy as np
from typing import List, Set, Tuple, Dict, Any
from catboost import CatBoostClassifier
import pickle
import os

class RankerL2:
    """L2-ранжирование, комбинирующее текстовые и векторные признаки с помощью CatBoost"""
    
    def __init__(self, ranker_l1, vector_index, 
                 model_path="models/L2_Ranker/model.pkl", 
                 model_text_only_path="models/L2_Ranker/model_text_only.pkl"):
        self.ranker_l1 = ranker_l1
        self.vector_index = vector_index
        self.model = None
        self.feature_names = []
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path, model_text_only_path)
            self.is_preloaded = True
        else:
            self.is_preloaded = False
            self.model = CatBoostClassifier(
                iterations=1000,
                depth=4,
                learning_rate=0.03,
                verbose=False,
                random_seed=42,
                task_type='CPU'
            )

            self.model_text_only = CatBoostClassifier(
                iterations=1000,
                depth=4,
                learning_rate=0.03,
                verbose=False,
                random_seed=42,
                task_type='CPU'
            )
            
    def get_features(self, query: str, candidates: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Получение матрицы признаков для кандидатов.
        
        Возвращает:
            X: матрица признаков [n_samples, n_features]
            feature_names: список названий признаков
        """
        features_list = []
        
        text_features = dict(self.ranker_l1.get_features(query, set(candidates)))
        vector_features = self.vector_index.get_maxP_for_candidates(query, candidates)
        
        for doc_id in candidates:
            doc_features = []
            feature_names = []
            
            text_feat = text_features.get(doc_id, [0.0, 0.0, 0.0, 0.0])
            doc_features.extend(text_feat)
            feature_names.extend(['overlap', 'tf_sum', 'prox', 'len_norm'])
            
            vector_score = vector_features.get(doc_id, 0.0)
            doc_features.append(vector_score)
            feature_names.append('maxp_score')
            
            
            features_list.append(doc_features)
            
            if not self.feature_names:
                self.feature_names = feature_names
        
        return np.array(features_list), self.feature_names
    
    def rank(self, query: str, text_candidates: List[Dict], 
             vector_candidates: List[Tuple[str, float]], 
             top_k: int = 50) -> List[str]:
        """Основной метод ранжирования L2.
        
        Args:
            query: поисковый запрос
            text_candidates: результаты текстового поиска [{'id': doc_id, 'score': score}]
            vector_candidates: результаты векторного поиска [(doc_id, score)]
            top_k: количество возвращаемых документов
            
        Returns:
            Список doc_id, отсортированный по убыванию релевантности
        """
        all_candidates = set()
        
        text_top = [cand['id'] for cand in text_candidates[:200]]
        vector_top = [doc_id for doc_id, _ in vector_candidates[:200]]
        
        all_candidates.update(text_top)
        all_candidates.update(vector_top)
        all_candidates = list(all_candidates)
        
        if not all_candidates:
            return []
        
        X, _  = self.get_features(query, all_candidates)
        scores = self.model.predict_proba(X)[:, 1]
        scores_text_only = self.model_text_only.predict_proba(X[:, :-1])[:, 1]


        scored_docs = list(zip(all_candidates, scores, scores_text_only))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return [{"id": doc_id, "score": score, "score_text_only": score_to} 
                for doc_id, score, score_to in scored_docs[:top_k]]
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray = None, y_val: np.ndarray = None):
        """Обучение модели CatBoost"""
        if X_val is not None and y_val is not None:
            self.model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                verbose=100,
                early_stopping_rounds=50
            )
            
            self.model_text_only.fit(
                X_train[:, :-1], y_train,
                eval_set=(X_val[:, :-1], y_val),
                verbose=100,
                early_stopping_rounds=50
            )
        else:
            self.model.fit(X_train, y_train, verbose=100)
            
            self.model_text_only.fit(X_train[:, :-1], y_train, verbose=100)
    
    def save_model(self, model_path: str):

        """Сохранение модели и метаданных"""
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names
        }
        with open(f"{model_path}/model.pkl", 'wb') as f:
            pickle.dump(model_data, f)

        model_data = {
            'model': self.model_text_only,
            'feature_names': self.feature_names
        }
        with open(f"{model_path}/model_text_only.pkl", 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str, model_text_only_path: str):
        """Загрузка модели и метаданных"""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        self.model = model_data['model']

        with open(model_text_only_path, 'rb') as f:
            model_data = pickle.load(f)
        self.model_text_only= model_data['model']

        
        self.feature_names = model_data['feature_names']
        print(f"Model loaded from {model_path}")