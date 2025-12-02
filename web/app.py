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

def load_documents_from_json(file_path: str = "~/search/wikipedia_ru_sample_500k.csv") -> List[JsonDocument]:
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

def initialize_documents():
    """Инициализация документов из JSON файла"""
    json_documents = load_documents_from_json()
    
    if not json_documents:
        # Fallback to sample data if JSON file is not available
        print("No documents found in JSON file, using sample data")
        initialize_sample_data()
        return
    
    for json_doc in tqdm(json_documents):
        doc = Document.create(
            title=json_doc.title,
            content=json_doc.text,
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

# Инициализация документов при запуске
initialize_documents()

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