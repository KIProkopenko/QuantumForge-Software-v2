"""
bot.py - RAG-бот QuantumForge Software с техниками Few-shot и Chain-of-Thought

FastAPI интерфейс для RAG-бота, который:
1. Ищет информацию в векторной базе знаний (FAISS)
2. Формирует промпт с Few-shot примерами и CoT инструкциями
3. Генерирует ответ через локальную LLM (Qwen2.5-1.5B-Instruct)

Автор: QuantumForge Software Team
"""

import os
import socket
import time

# 🔧 Принудительное использование IPv4 (для VPN)
_original_getaddrinfo = socket.getaddrinfo
def _ipv4_only_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    return _original_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)
socket.getaddrinfo = _ipv4_only_getaddrinfo

# 🔧 OFFLINE режим — не обращаться к сети
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

from typing import List, Dict, Any, Optional

# === LANGCHAIN IMPORTS (версия 1.x) ===
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# === FASTAPI ===
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# === TRANSFORMERS ===
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# === НАШИ ПРОМПТЫ С FEW-SHOT И COT ===
from prompts import SYSTEM_PROMPT, FEW_SHOT_EXAMPLES, build_prompt, format_context


# ==================== МОДЕЛИ ДАННЫХ ====================

class QueryRequest(BaseModel):
    question: str
    k: int = 3  # Количество чанков для поиска

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    elapsed_seconds: float

class HealthResponse(BaseModel):
    status: str
    vectorstore_loaded: bool
    model_loaded: bool
    documents_count: int
    chunks_count: int


# ==================== ИНИЦИАЛИЗАЦИЯ FASTAPI ====================

app = FastAPI(
    title="QuantumForge RAG Bot",
    description="RAG-бот с техниками Few-shot и Chain-of-Thought",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ ====================

vectorstore = None
qa_chain = None
llm = None
embeddings = None
documents_count = 0
chunks_count = 0


# ==================== КОНФИГУРАЦИЯ ====================

class Config:
    KB_PATH = "knowledge_base"
    PERSIST_DIRECTORY = "faiss_index"
    
    # Локальные пути к моделям
    EMBEDDING_MODEL = os.path.expanduser(
        "~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    )
    LLM_MODEL = os.path.expanduser(
        "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
    )
    
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    MAX_NEW_TOKENS = 512
    TEMPERATURE = 0.3
    DEVICE = "cpu"

config = Config()


# ==================== ЗАГРУЗКА ДОКУМЕНТОВ ====================

def load_documents(kb_path: str = None) -> List[Any]:
    """Загрузка документов из knowledge_base"""
    kb_path = kb_path or config.KB_PATH
    print(f"\n📂 Загрузка документов из {kb_path}...")
    
    if not os.path.exists(kb_path):
        print(f"❌ Папка {kb_path} не найдена!")
        return []
    
    loader = DirectoryLoader(
        kb_path,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    documents = loader.load()
    print(f"✅ Загружено {len(documents)} документов")
    
    # Добавляем метаданные
    for i, doc in enumerate(documents):
        source_path = doc.metadata.get("source", "unknown")
        doc.metadata["doc_id"] = i
        doc.metadata["filename"] = os.path.basename(source_path)
        doc.metadata["title"] = os.path.splitext(os.path.basename(source_path))[0].replace("_", " ")
    
    return documents


def split_documents(documents: List[Any]) -> List[Any]:
    """Разбиение документов на чанки"""
    print(f"\n✂️  Разбиение документов на чанки...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    
    # Добавляем chunk_id
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    
    print(f"✅ Создано {len(chunks)} чанков")
    return chunks


# ==================== ВЕКТОРНОЕ ХРАНИЛИЩЕ ====================

def get_embeddings():
    """Загрузка модели эмбеддингов"""
    global embeddings
    if embeddings is None:
        print(f"\n🧠 Загрузка модели эмбеддингов (локально)...")
        embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={'device': config.DEVICE},
            encode_kwargs={'normalize_embeddings': False}
        )
        print(f"✅ Модель: all-MiniLM-L6-v2 (384-мерные векторы)")
    return embeddings


def create_vectorstore(chunks: List[Any], persist_directory: str = None) -> FAISS:
    """Создание FAISS индекса"""
    persist_directory = persist_directory or config.PERSIST_DIRECTORY
    print(f"\n🗄️  Создание векторного хранилища FAISS...")
    
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    os.makedirs(persist_directory, exist_ok=True)
    vectorstore.save_local(persist_directory)
    print(f"✅ Индекс сохранён в {persist_directory}/")
    return vectorstore


def load_vectorstore(persist_directory: str = None) -> Optional[FAISS]:
    """Загрузка существующего FAISS индекса"""
    persist_directory = persist_directory or config.PERSIST_DIRECTORY
    if not os.path.exists(persist_directory):
        return None
    
    print(f"\n📥 Загрузка FAISS индекса из {persist_directory}/...")
    embeddings = get_embeddings()
    vectorstore = FAISS.load_local(
        persist_directory,
        embeddings,
        allow_dangerous_deserialization=True
    )
    print(f"✅ Индекс загружен")
    return vectorstore


# ==================== LLM МОДЕЛЬ ====================

def load_llm():
    """Загрузка LLM модели"""
    global llm
    if llm is not None:
        return llm
    
    print(f"\n🤖 Загрузка LLM модели (локально)...")
    print(f"   Модель: Qwen2.5-1.5B-Instruct")
    
    tokenizer = AutoTokenizer.from_pretrained(
        config.LLM_MODEL,
        trust_remote_code=True,
        local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.LLM_MODEL,
        device_map="cpu",
        trust_remote_code=True,
        local_files_only=True
    )
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=config.MAX_NEW_TOKENS,
        temperature=config.TEMPERATURE,
        repetition_penalty=1.1,
        return_full_text=False,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    llm = HuggingFacePipeline(pipeline=pipe)
    print(f"✅ LLM модель загружена")
    return llm


# ==================== QA CHAIN С FEW-SHOT И COT ====================

def create_qa_chain(vectorstore: FAISS) -> RetrievalQA:
    print(f"\n🔗 Создание QA цепочки с Few-shot и CoT...")
    llm = load_llm()
    
    # Chat-формат для Qwen2.5
    prompt_template = (
        "<|im_start|>system\n"
        "You are a helpful assistant. Answer questions based ONLY on the provided context.\n"
        "ALWAYS think step by step and show your reasoning.\n"
        "Format: 'Let me think step by step: 1... 2... 3... Therefore...'\n"
        "If answer is not in context, say 'I don't know'.<|im_end|>\n"
        "<|im_start|>user\n"
        f"{FEW_SHOT_EXAMPLES}\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}<|im_end|>\n"
        "<|im_start|>assistant\n"
        "Let me think step by step:"
    )
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    print(f"✅ QA цепочка готова (Few-shot + CoT)")
    return qa_chain


# ==================== ИНИЦИАЛИЗАЦИЯ БОТА ====================

def initialize_bot():
    """Полная инициализация бота"""
    global vectorstore, qa_chain, documents_count, chunks_count
    
    print("\n" + "="*60)
    print("🤖 ИНИЦИАЛИЗАЦИЯ RAG-БОТА")
    print("="*60)
    
    # Загрузка или создание индекса
    if os.path.exists(config.PERSIST_DIRECTORY):
        vectorstore = load_vectorstore(config.PERSIST_DIRECTORY)
        if vectorstore is None:
            documents = load_documents(config.KB_PATH)
            if not documents:
                print("❌ Нет документов для индексации!")
                return False
            chunks = split_documents(documents)
            vectorstore = create_vectorstore(chunks, config.PERSIST_DIRECTORY)
            documents_count = len(documents)
            chunks_count = len(chunks)
    else:
        documents = load_documents(config.KB_PATH)
        if not documents:
            print("❌ Нет документов для индексации!")
            return False
        chunks = split_documents(documents)
        vectorstore = create_vectorstore(chunks, config.PERSIST_DIRECTORY)
        documents_count = len(documents)
        chunks_count = len(chunks)
    
    # Создание QA цепочки
    qa_chain = create_qa_chain(vectorstore)
    
    print("\n" + "="*60)
    print("✅ БОТ ГОТОВ К РАБОТЕ!")
    print(f"📚 Документов: {documents_count}")
    print(f"📑 Чанков: {chunks_count}")
    print(f"🧠 Эмбеддинги: all-MiniLM-L6-v2")
    print(f"🤖 LLM: Qwen2.5-1.5B-Instruct")
    print(f"💡 Техники: Few-shot + Chain-of-Thought")
    print("="*60 + "\n")
    return True


# ==================== FASTAPI ENDPOINTS ====================

@app.on_event("startup")
async def startup_event():
    """Событие при запуске приложения"""
    success = initialize_bot()
    if not success:
        print("⚠️  Бот запущен, но не все компоненты инициализированы")


@app.get("/")
async def root():
    """Корневой эндпоинт"""
    return {
        "message": "QuantumForge RAG Bot is running",
        "version": "2.0.0",
        "features": ["Few-shot prompting", "Chain-of-Thought", "FAISS search"],
        "endpoints": {
            "/": "GET - This info",
            "/query": "POST - Ask a question",
            "/health": "GET - Health check",
            "/docs": "GET - Swagger UI documentation"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Проверка здоровья"""
    return HealthResponse(
        status="healthy",
        vectorstore_loaded=vectorstore is not None,
        model_loaded=llm is not None,
        documents_count=documents_count,
        chunks_count=chunks_count
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Обработка вопроса пользователя.
    
    Пайплайн:
    1. Поиск релевантных чанков в FAISS
    2. Формирование промпта с Few-shot и CoT
    3. Генерация ответа через LLM
    """
    if qa_chain is None:
        raise HTTPException(
            status_code=500, 
            detail="Bot not initialized. Please wait for initialization."
        )
    
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        start_time = time.time()
        print(f"\n❓ Вопрос: {request.question}")
        
        # Выполняем RAG-пайплайн
        result = qa_chain.invoke({"query": request.question})
        
        # Извлекаем источники с метаданными
        sources = []
        if "source_documents" in result:
            for doc in result["source_documents"]:
                sources.append({
                    "filename": doc.metadata.get("filename", "unknown"),
                    "title": doc.metadata.get("title", "unknown"),
                    "chunk_id": doc.metadata.get("chunk_id", "N/A"),
                    "content_preview": doc.page_content[:200] + "..."
                })
        
        answer = result.get("result", "No answer generated")
        elapsed = time.time() - start_time
        
        print(f"💬 Ответ: {answer[:100]}...")
        print(f"📚 Источники: {len(sources)}")
        print(f"⏱️  Время: {elapsed:.2f} сек")
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            elapsed_seconds=round(elapsed, 2)
        )
    
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search")
async def search_only(request: QueryRequest):
    """
    Только поиск по базе (без генерации ответа).
    Полезно для отладки и анализа качества поиска.
    """
    if vectorstore is None:
        raise HTTPException(status_code=500, detail="Vectorstore not initialized")
    
    try:
        results = vectorstore.similarity_search_with_score(
            request.question, 
            k=request.k
        )
        
        sources = []
        for doc, score in results:
            sources.append({
                "filename": doc.metadata.get("filename", "unknown"),
                "title": doc.metadata.get("title", "unknown"),
                "chunk_id": doc.metadata.get("chunk_id", "N/A"),
                "score": float(score),
                "similarity": 1 / (1 + score),
                "content": doc.page_content
            })
        
        return {"query": request.question, "results": sources}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/rebuild-index")
async def rebuild_index():
    """Перестроить индекс (для администрирования)"""
    global vectorstore, qa_chain
    
    try:
        print("🔄 Перестройка индекса...")
        
        if os.path.exists(config.PERSIST_DIRECTORY):
            import shutil
            shutil.rmtree(config.PERSIST_DIRECTORY)
            print(f"Удалён старый индекс из {config.PERSIST_DIRECTORY}")
        
        documents = load_documents(config.KB_PATH)
        chunks = split_documents(documents)
        vectorstore = create_vectorstore(chunks, config.PERSIST_DIRECTORY)
        qa_chain = create_qa_chain(vectorstore)
        
        return {
            "status": "success",
            "message": "Index rebuilt successfully",
            "documents": len(documents),
            "chunks": len(chunks)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error rebuilding index: {str(e)}")


# ==================== ЗАПУСК ====================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 QUANTUMFORGE RAG BOT v2.0")
    print("="*60)
    print(f"📚 Knowledge Base: {config.KB_PATH}")
    print(f"🗄️  Vector Store: {config.PERSIST_DIRECTORY}")
    print(f"🧠 Embeddings: all-MiniLM-L6-v2 (384-dim)")
    print(f"🤖 LLM: Qwen2.5-1.5B-Instruct")
    print(f"💡 Techniques: Few-shot + Chain-of-Thought")
    print("="*60)
    
    print("\n🌐 Запуск API сервера на http://localhost:8000")
    print("📚 Swagger UI: http://localhost:8000/docs")
    print("💚 Health check: http://localhost:8000/health")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")