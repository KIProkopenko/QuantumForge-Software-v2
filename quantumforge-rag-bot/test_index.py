#!/usr/bin/env python3
"""
test_index.py - Тестирование векторного индекса FAISS

Этот скрипт:
1. Загружает существующий FAISS индекс
2. Выполняет тестовые запросы
3. Показывает найденные чанки с метаданными
"""

import os
import socket

# IPv4 only
_original_getaddrinfo = socket.getaddrinfo
def _ipv4_only_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    return _original_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)
socket.getaddrinfo = _ipv4_only_getaddrinfo

# OFFLINE режим
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ==================== КОНФИГУРАЦИЯ ====================

INDEX_DIR = "faiss_index"
EMBEDDING_MODEL = os.path.expanduser(
    "~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
)


# ==================== ТЕСТОВЫЕ ЗАПРОСЫ ====================

TEST_QUERIES = [
    # Запросы, на которые бот ДОЛЖЕН ответить (из базы знаний)
    "Who is Xarn Velgor?",
    "What is the Void Core?",
    "Tell me about Kael Torren",
    
    # Запросы, на которые бот НЕ должен находить релевантных документов
    "Who is Darth Vader?",
    "What is the Death Star?",
]


# ==================== ФУНКЦИИ ====================

def load_index():
    """Загрузка существующего FAISS индекса"""
    print(f"\n{'='*60}")
    print(f"Загрузка FAISS индекса из {INDEX_DIR}")
    print(f"{'='*60}")
    
    if not os.path.exists(INDEX_DIR):
        print(f"Индекс не найден в {INDEX_DIR}")
        print("Сначала запустите: python build_index.py")
        return None
    
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )
    
    vectorstore = FAISS.load_local(
        INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    print(f"Индекс загружен")
    return vectorstore


def test_query(vectorstore, query, k=3):
    """Тестирование одного запроса"""
    print(f"\n{'='*60}")
    print(f"ЗАПРОС: {query}")
    print(f"{'='*60}")
    
    # Поиск похожих чанков
    results = vectorstore.similarity_search_with_score(query, k=k)
    
    print(f"Найдено {len(results)} релевантных чанков:\n")
    
    for i, (doc, score) in enumerate(results, 1):
        # similarity_search_with_score возвращает L2 distance (меньше = лучше)
        similarity = 1 / (1 + score)
        
        print(f"   [{i}] Score: {score:.4f} (similarity: {similarity:.4f})")
        print(f"   Источник: {doc.metadata.get('filename', 'unknown')}")
        print(f"   Заголовок: {doc.metadata.get('title', 'unknown')}")
        print(f"   Chunk ID: {doc.metadata.get('chunk_id', 'N/A')}")
        print(f"   Текст:")
        # Показываем первые 300 символов
        text_preview = doc.page_content[:300].replace('\n', ' ')
        print(f"      {text_preview}...")
        print()
    
    return results


def run_tests(vectorstore):
    """Запуск всех тестовых запросов"""
    print(f"\n{'#'*60}")
    print(f"ТЕСТИРОВАНИЕ ВЕКТОРНОГО ИНДЕКСА")
    print(f"{'#'*60}")
    
    all_results = {}
    
    for query in TEST_QUERIES:
        results = test_query(vectorstore, query)
        all_results[query] = results
    
    # Итоговая статистика
    print(f"\n{'#'*60}")
    print(f"ИТОГИ ТЕСТИРОВАНИЯ")
    print(f"{'#'*60}")
    
    print("\nЗапросы из базы знаний (должны найти релевантные документы):")
    for query in TEST_QUERIES[:3]:
        results = all_results[query]
        top_score = results[0][1] if results else float('inf')
        status = "OK" if top_score < 1.0 else "WARN"
        print(f"   [{status}] '{query}' - top score: {top_score:.4f}")
    
    print("\nЗапросы НЕ из базы знаний:")
    for query in TEST_QUERIES[3:]:
        results = all_results[query]
        top_score = results[0][1] if results else float('inf')
        status = "WARN" if top_score < 1.5 else "OK"
        print(f"   [{status}] '{query}' - top score: {top_score:.4f}")
    
    print("\nПримечание:")
    print("   - Score < 1.0: высокая релевантность")
    print("   - Score 1.0-1.5: средняя релевантность")
    print("   - Score > 1.5: низкая релевантность")


# ==================== MAIN ====================

def main():
    print("\n" + "="*60)
    print("TEST INDEX - QuantumForge Knowledge Base")
    print("="*60)
    
    # Загрузка индекса
    vectorstore = load_index()
    if vectorstore is None:
        return
    
    # Запуск тестов
    run_tests(vectorstore)
    
    print("\n" + "="*60)
    print("Тестирование завершено!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()