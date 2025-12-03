from flask import Flask, render_template, request, jsonify
import sys
import os
import json
from dataclasses import dataclass
from typing import List, Dict, Optional
from tqdm import tqdm
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.index import SearchEngine
from core.document import Document

app = Flask(__name__)
search_engine = SearchEngine()

@dataclass
class JsonDocument:
    title: str
    text: str

@dataclass
class MsMarcoDocument:
    title: str
    body: str
    url: str
    ms_marco_id: str

def load_documents_from_json(file_path: str = "./datasets/wikipedia_ru_sample_500k.csv") -> List[JsonDocument]:
    """Загрузка документов из JSON файла"""
    try:
        data = pd.read_csv(file_path)
        documents = []
        for title, text in zip(list(data.title), list(data.text)):
            documents.append(JsonDocument(
                title=title,
                text=text
            ))
        
        print(f"Loaded {len(documents)} documents from {file_path}")
        return documents
    
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return []
    except Exception as e:
        print(f"Error loading documents: {e}")
        return []

def load_ms_marco_documents(file_path: str = "./datasets/documents_train.csv") -> List[MsMarcoDocument]:
    """Загрузка документов из MSMARCO"""
    try:
        data = pd.read_csv(file_path)[:10000]
        documents = []
        for title, body, url, doc_id  in zip(list(data.title), list(data.body), data.url, data.doc_id):
            documents.append(MsMarcoDocument(
                title=title,
                body=body,
                url=url,
                ms_marco_id=doc_id,
            ))
        
        print(f"Loaded {len(documents)} documents from {file_path}")
        return documents
    
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return []
    except Exception as e:
        print(f"Error loading documents: {e}")
        return []

def initialize_documents():
    """Инициализация документов из JSON файла"""
    json_documents = load_ms_marco_documents()
    
    if not json_documents:
        # Fallback to sample data if JSON file is not available
        print("No documents found in JSON file, using sample data")
        initialize_sample_data()
        return
    
    for json_doc in tqdm(json_documents):
        doc = Document.create(
            title=json_doc.title,
            content=json_doc.body,
            url=json_doc.url,
            ms_marco_id=json_doc.ms_marco_id,
            author="Unknown"  # Default author since JSON doesn't have this field
        )
        search_engine.add_document(doc)
    search_engine.inverted_index.flush()
    
    print(f"Initialized {len(json_documents)} documents in search engine")

def initialize_sample_data():
    """Резервная инициализация тестовыми данными"""
    sample_docs = [
        Document.create(
            title="Python Programming", 
            content="Python is a great programming language for web development and data science.",
            author="John Doe"
        ),
        Document.create(
            title="Web Development",
            content="Modern web development involves Python, JavaScript, and other technologies.",
            author="Jane Smith"
        ),
        Document.create(
            title="Data Science",
            content="Data science uses Python for machine learning and data analysis.",
            author="John Doe"
        ),
        Document.create(
            title="Machine Learning",
            content="Python is popular for machine learning and artificial intelligence projects.",
            author="Bob Wilson"
        )
    ]
    
    for doc in sample_docs:
        search_engine.add_document(doc)

def learn_l1_ranker():
    try:
        qrels = pd.read_csv("datasets/queries_train.csv", index_col=0)
        n_training_sample = 5000
        # qrels = qrels.sample(n_training_sample)

        
        print(f"Загружено {len(qrels)} запросов для обучения")

        ms_marco_id_to_local_id_mapping = {}


        for doc_id, doc in search_engine.inverted_index.documents.items():
            ms_marco_id_to_local_id_mapping[doc.fields.get('ms_marco_id')] = doc_id

        indices = []
        for i in range(n_training_sample):
            if qrels.doc_id[i] in ms_marco_id_to_local_id_mapping:
                indices.append(i)

        print(f"Отфильтровано {len(qrels) - len(indices)} ({len(indices)/len(qrels)}%) запросов")
        qrels = qrels.iloc[indices]
        
        X = []
        y = []        

        for _, row in tqdm(qrels.iterrows(), total=len(qrels), desc="Настреливаем поиск за фичами и кандидатами"):
            query_text = row['text']
            relevant_doc_id = row['doc_id']
            try:
                candidates = search_engine.get_candidates_and_features(query_text)
                true_docs = search_engine.ranker_l1.get_features(query_text, [ms_marco_id_to_local_id_mapping[relevant_doc_id]])
            except:
                print("Bad query:", query_text)
                continue
            candidates = candidates + true_docs

            if not candidates:
                continue
                
            for doc_id, features in candidates:

                ms_marco_id = search_engine.inverted_index.documents.get(doc_id).fields.get('ms_marco_id', 'No title')
                label = 1 if ms_marco_id == relevant_doc_id else 0
                
                X.append(features)
                y.append(label)
                
        print(f"Собрано {len(X)} примеров для обучения")
        print(f"Распределение классов: {sum(y)} положительных, {len(y)-sum(y)} отрицательных")
        
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import classification_report, accuracy_score
            
            import numpy as np
            X_np = np.array(X)
            y_np = np.array(y)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_np, y_np, test_size=0.2, random_state=42, stratify=y_np
            )
            
            model = LogisticRegression(
                max_iter=1000,  # Увеличиваем итерации для сходимости
                random_state=42,
                class_weight='balanced'  # Балансировка классов
            )
            
            print("Обучаем логистическую регрессию...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            print("\n=== Веса модели ===")
            for i, weight in enumerate(model.coef_[0]):
                print(f"Признак {i}: {weight:.6f}")
            
            print(f"Свободный член (intercept): {model.intercept_[0]:.6f}")
            
            # Сохраняем веса модели в файл
            model_info = {
                'weights': model.coef_[0].tolist(),
                'intercept': model.intercept_[0].item(),
                'accuracy': accuracy_score(y_test, y_pred)
            }
            search_engine.ranker_l1.weights = model.coef_[0].tolist() + [model.intercept_[0].item()]
            import json
            with open('l1_ranker_weights.json', 'w') as f:
                json.dump(model_info, f, indent=2)
            
            print("\nВеса модели сохранены в файл 'l1_ranker_weights.json'")
            return model
            
        except ImportError as e:
            print(f"Ошибка импорта библиотек: {e}")
            print("Установите scikit-learn: pip install scikit-learn")
            return None
            
    except Exception as e:
        print(f"Ошибка при обучении L1 ранкера: {e}")
        import traceback
        traceback.print_exc()
        return None



initialize_documents()
learn_l1_ranker()

@app.route('/')
def index():
    fields = list(search_engine.get_available_fields())
    doc_count = len(search_engine.inverted_index.documents)
    return render_template('index.html', fields=fields, doc_count=doc_count)

@app.route('/search')
def search():
    query = request.args.get('q', '')
    if not query:
        return render_template('results.html', query=query, results=[])

    try:
        documents = search_engine.search(query)
        doc_ids = [r['id'] for r in documents]
        results = []
        for doc_id in doc_ids:
            doc = search_engine.inverted_index.documents.get(doc_id)
            if doc:
                content = doc.fields.get('content', '')
                preview_length = 200  # Количество символов для предпросмотра
                
                results.append({
                    'id': doc.id,
                    'title': doc.fields.get('title', 'No title'),
                    'content': content,
                    'author': doc.fields.get('author', 'Unknown'),
                    'preview': (content[:preview_length] + '...' if len(content) > preview_length 
                                else content)
                })
        
        return render_template('results.html', query=query, results=results[:100])
    
    except Exception as e:
        return render_template('results.html', query=query, error=str(e), results=[])

if __name__ == '__main__':
    app.run(use_reloader=False, debug=True, port=5000)