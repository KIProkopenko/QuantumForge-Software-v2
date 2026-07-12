#!/usr/bin/env python3
"""
build_index.py - Создание векторного индекса FAISS для базы знаний QuantumForge

Этот скрипт:
1. Загружает документы из knowledge_base/
2. Разбивает их на чанки
3. Генерирует эмбеддинги с помощью all-MiniLM-L6-v2
4. Создаёт и сохраняет FAISS индекс
"""

import os
import socket
import time

# Принудительное использование IPv4 (для VPN)
_original_getaddrinfo = socket.getaddrinfo
def _ipv4_only_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    return _original_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)
socket.getaddrinfo = _ipv4_only_getaddrinfo

# OFFLINE режим
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ==================== КОНФИГУРАЦИЯ ====================

class IndexConfig:
    # Пути
    KB_PATH = "knowledge_base"
    INDEX_DIR = "faiss_index"
    
    # Модель эмбеддингов (локальный путь)
    EMBEDDING_MODEL = os.path.expanduser(
        "~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    )
    
    # Параметры разбиения на чанки
    CHUNK_SIZE = 1000        # символов (~250 слов)
    CHUNK_OVERLAP = 200      # символов перекрытия
    
    # Устройство (CPU для совместимости)
    DEVICE = "cpu"

config = IndexConfig()


# ==================== ФУНКЦИИ ====================

def load_documents():
    """Загрузка документов из knowledge_base"""
    print(f"\n{'='*60}")
    print(f"Загрузка документов из {config.KB_PATH}")
    print(f"{'='*60}")
    
    loader = DirectoryLoader(
        config.KB_PATH,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    
    documents = loader.load()
    print(f"Загружено документов: {len(documents)}")
    
    # Добавляем метаданные к каждому документу
    for i, doc in enumerate(documents):
        source_path = doc.metadata.get("source", "unknown")
        doc.metadata["doc_id"] = i
        doc.metadata["filename"] = os.path.basename(source_path)
        doc.metadata["title"] = os.path.splitext(os.path.basename(source_path))[0].replace("_", " ")
    
    return documents


def split_documents(documents):
    """Разбиение документов на чанки"""
    print(f"\n{'='*60}")
    print(f"Разбиение документов на чанки")
    print(f"{'='*60}")
    print(f"   Chunk size: {config.CHUNK_SIZE} символов")
    print(f"   Chunk overlap: {config.CHUNK_OVERLAP} символов")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    
    # Добавляем ID чанка в метаданные
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    
    print(f"Создано чанков: {len(chunks)}")
    return chunks


def create_embeddings():
    """Загрузка модели эмбеддингов"""
    print(f"\n{'='*60}")
    print(f"Загрузка модели эмбеддингов")
    print(f"{'='*60}")
    print(f"   Модель: all-MiniLM-L6-v2")
    print(f"   Путь: {config.EMBEDDING_MODEL}")
    print(f"   Размерность: 384")
    print(f"   Устройство: {config.DEVICE}")
    
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={'device': config.DEVICE},
        encode_kwargs={'normalize_embeddings': False}
    )
    
    print(f"Модель загружена")
    return embeddings


def create_index(chunks, embeddings):
    """Создание FAISS индекса"""
    print(f"\n{'='*60}")
    print(f"Создание FAISS индекса")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # Создаём индекс
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Сохраняем на диск
    os.makedirs(config.INDEX_DIR, exist_ok=True)
    vectorstore.save_local(config.INDEX_DIR)
    
    elapsed = time.time() - start_time
    
    print(f"Индекс создан и сохранён в {config.INDEX_DIR}/")
    print(f"Время создания: {elapsed:.2f} сек")
    
    return vectorstore, elapsed


def print_stats(documents, chunks, elapsed):
    """Вывод статистики"""
    print(f"\n{'='*60}")
    print(f"СТАТИСТИКА ИНДЕКСАЦИИ")
    print(f"{'='*60}")
    print(f"   Модель эмбеддингов: all-MiniLM-L6-v2")
    print(f"   Ссылка: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2")
    print(f"   Размерность эмбеддингов: 384")
    print(f"   Количество документов: {len(documents)}")
    print(f"   Количество чанков: {len(chunks)}")
    print(f"   Размер чанка: {config.CHUNK_SIZE} символов")
    print(f"   Перекрытие: {config.CHUNK_OVERLAP} символов")
    print(f"   Время генерации: {elapsed:.2f} сек")
    print(f"   Директория индекса: {config.INDEX_DIR}/")
    print(f"{'='*60}\n")


# ==================== MAIN ====================

def main():
    print("\n" + "="*60)
    print("BUILD INDEX - QuantumForge Knowledge Base")
    print("="*60)
    
    # Проверка наличия knowledge_base
    if not os.path.exists(config.KB_PATH):
        print(f"Папка {config.KB_PATH} не найдена!")
        print("Создайте папку и добавьте .md файлы")
        return
    
    # 1. Загрузка документов
    documents = load_documents()
    if not documents:
        print("Нет документов для индексации!")
        return
    
    # 2. Разбиение на чанки
    chunks = split_documents(documents)
    
    # 3. Загрузка модели эмбеддингов
    embeddings = create_embeddings()
    
    # 4. Создание индекса
    vectorstore, elapsed = create_index(chunks, embeddings)
    
    # 5. Вывод статистики
    print_stats(documents, chunks, elapsed)
    
    print("Индекс успешно создан!")
    print(f"Файлы индекса находятся в: {config.INDEX_DIR}/")
    print("\nДля тестирования запустите:")
    print("   python test_index.py")


if __name__ == "__main__":
    main()