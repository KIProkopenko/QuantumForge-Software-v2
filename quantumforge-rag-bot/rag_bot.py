"""
rag_bot.py - Основной модуль RAG-бота

Этот модуль реализует полный RAG-пайплайн:
1. Загрузка FAISS индекса
2. Преобразование запроса в эмбеддинг
3. Поиск релевантных чанков
4. Формирование промпта с Few-shot и CoT
5. Генерация ответа через LLM
6. Защита от промпт-инъекций (3 уровня)

Автор: QuantumForge Software Team
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

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from prompts import build_prompt, format_context


# ==================== КОНФИГУРАЦИЯ ====================

class RAGConfig:
    """Конфигурация RAG-бота."""

    # Пути к локальным моделям
    EMBEDDING_MODEL = os.path.expanduser(
        "~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    )
    LLM_MODEL = os.path.expanduser(
        "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
    )

    # FAISS индекс
    INDEX_DIR = "faiss_index"

    # Параметры поиска
    TOP_K = 3

    # Параметры генерации
    MAX_NEW_TOKENS = 512
    TEMPERATURE = 0.3

    DEVICE = "cpu"


config = RAGConfig()


# ==================== КЛАСС RAG BOT ====================

class RAGBot:
    """
    RAG-бот с техниками Few-shot, Chain-of-Thought и защитой от инъекций.

    Атрибуты:
        embeddings: модель эмбеддингов для векторизации запросов
        vectorstore: FAISS индекс с базой знаний
        llm: языковая модель для генерации ответов
    """

    def __init__(self):
        """Инициализация RAG-бота: загрузка моделей и индекса."""
        print("\n" + "=" * 60)
        print("ИНИЦИАЛИЗАЦИЯ RAG-БОТА")
        print("=" * 60)

        self.embeddings = None
        self.vectorstore = None
        self.llm = None

        self._load_embeddings()
        self._load_index()
        self._load_llm()

        print("=" * 60)
        print("БОТ ГОТОВ К РАБОТЕ!")
        print("=" * 60 + "\n")

    def _load_embeddings(self):
        """Загрузка модели эмбеддингов."""
        print("\n[1/3] Загрузка модели эмбеддингов...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={'device': config.DEVICE},
            encode_kwargs={'normalize_embeddings': False}
        )
        print("   Модель: all-MiniLM-L6-v2 (384-мерные векторы)")

    def _load_index(self):
        """Загрузка FAISS индекса."""
        print("\n[2/3] Загрузка FAISS индекса...")
        if not os.path.exists(config.INDEX_DIR):
            raise FileNotFoundError(
                f"Индекс не найден в {config.INDEX_DIR}. "
                "Сначала запустите: python build_index.py"
            )

        self.vectorstore = FAISS.load_local(
            config.INDEX_DIR,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        print(f"   Индекс загружен из {config.INDEX_DIR}/")

    def _load_llm(self):
        """Загрузка LLM модели."""
        print("\n[3/3] Загрузка LLM модели...")
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

        self.llm = HuggingFacePipeline(pipeline=pipe)
        print(f"   Модель: Qwen2.5-1.5B-Instruct")

    def search(self, query: str, k: int = None) -> list:
        """
        Поиск релевантных чанков в векторной базе.

        Args:
            query: Поисковый запрос
            k: Количество результатов (по умолчанию TOP_K)

        Returns:
            Список кортежей (документ, score)
        """
        k = k or config.TOP_K
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        return results

    def answer(self, question: str) -> dict:
        """
        Полный RAG-пайплайн: поиск + фильтрация + промптинг + генерация + пост-проверка.

        Реализует три уровня защиты от промпт-инъекций:
        1. Pre-question проверка — блокировка инъекций в самом вопросе
        2. Pre-retrieval фильтрация — удаление вредоносных чанков
        3. Post-generation проверка — анализ ответа на утечки

        Args:
            question: Вопрос пользователя

        Returns:
            Словарь с ответом, источниками, логами безопасности и временем выполнения
        """
        from safety import (
            filter_chunks,
            sanitize_output,
            is_injection_attempt
        )

        start_time = time.time()
        safety_log = []

        # УРОВЕНЬ 1: Проверка вопроса на инъекцию
        print(f"\nЗапрос: '{question}'")

        if is_injection_attempt(question):
            print("   Обнаружена попытка промпт-инъекции в вопросе!")
            safety_log.append({
                "level": "pre-question",
                "action": "blocked",
                "reason": "injection_attempt_in_question"
            })
            return {
                "question": question,
                "answer": "I detected a prompt injection attempt in your question. I cannot comply with requests to ignore my instructions or reveal sensitive information.",
                "sources": [],
                "safety_log": safety_log,
                "elapsed_seconds": round(time.time() - start_time, 2)
            }

        # ШАГ 1: Поиск релевантных чанков
        search_results = self.search(question)
        documents = [doc for doc, score in search_results]

        sources = []
        for doc, score in search_results:
            sources.append({
                "filename": doc.metadata.get("filename", "unknown"),
                "title": doc.metadata.get("title", "unknown"),
                "score": float(score),
                "similarity": 1 / (1 + score)
            })

        print(f"   Найдено {len(documents)} релевантных чанков")

        # УРОВЕНЬ 2: Фильтрация вредоносных чанков
        safe_documents, filtered_info = filter_chunks(documents)

        if filtered_info:
            print(f"   Отфильтровано {len(filtered_info)} вредоносных/конфиденциальных чанков:")
            for info in filtered_info:
                print(f"      - {info['filename']}: {info['reason']}")
                safety_log.append({
                    "level": "pre-retrieval",
                    "action": "filtered",
                    "filename": info["filename"],
                    "reason": info["reason"]
                })

        if not safe_documents:
            print("   Все найденные чанки отфильтрованы!")
            return {
                "question": question,
                "answer": "I don't know. All retrieved documents were filtered due to safety concerns.",
                "sources": sources,
                "safety_log": safety_log,
                "filtered_chunks": len(filtered_info),
                "elapsed_seconds": round(time.time() - start_time, 2)
            }

        # ШАГ 2: Формирование контекста из БЕЗОПАСНЫХ чанков
        context = format_context(safe_documents)

        # ШАГ 3: Построение промпта с Few-shot и CoT
        prompt = build_prompt(question, context)
        print(f"   Промпт сформирован (Few-shot + CoT + Safety)")

        # ШАГ 4: Генерация ответа через LLM
        print(f"   Генерация ответа через LLM...")
        answer_text = self.llm.invoke(prompt)

        # УРОВЕНЬ 3: Пост-проверка ответа
        sanitized_answer, was_modified, leaked = sanitize_output(answer_text)

        if was_modified:
            print(f"   ВНИМАНИЕ: Обнаружена утечка конфиденциальных данных в ответе!")
            print(f"      Найденные паттерны: {leaked}")
            print(f"      Ответ был очищен.")
            safety_log.append({
                "level": "post-generation",
                "action": "sanitized",
                "leaked_patterns": leaked
            })

        elapsed = time.time() - start_time

        return {
            "question": question,
            "answer": sanitized_answer.strip(),
            "sources": sources,
            "safety_log": safety_log,
            "filtered_chunks": len(filtered_info),
            "elapsed_seconds": round(elapsed, 2)
        }

    def chat(self):
        """Интерактивный режим (REPL) с отображением логов безопасности."""
        print("\n" + "=" * 60)
        print("ИНТЕРАКТИВНЫЙ РЕЖИМ RAG-БОТА")
        print("=" * 60)
        print("Введите вопрос (или 'exit' для выхода):\n")

        while True:
            try:
                question = input("Вопрос: ").strip()

                if question.lower() in ['exit', 'quit', 'выход']:
                    print("\nДо свидания!")
                    break

                if not question:
                    continue

                result = self.answer(question)

                print("\n" + "-" * 60)
                print("ОТВЕТ БОТА:")
                print("-" * 60)
                print(result["answer"])
                print("-" * 60)

                print(f"\nИсточники ({len(result['sources'])}):")
                for i, src in enumerate(result["sources"], 1):
                    print(f"   [{i}] {src['filename']} (score: {src['score']:.4f})")

                if result.get("safety_log"):
                    print(f"\nЛоги безопасности:")
                    for log in result["safety_log"]:
                        print(f"   - [{log['level']}] {log['action']}: {log.get('reason', '')}")

                print(f"\nВремя: {result['elapsed_seconds']} сек")
                print("=" * 60 + "\n")

            except KeyboardInterrupt:
                print("\n\nДо свидания!")
                break
            except Exception as e:
                print(f"\nОшибка: {e}\n")


# ==================== MAIN ====================

def main():
    """Точка входа для запуска бота."""
    bot = RAGBot()
    bot.chat()


if __name__ == "__main__":
    main()