FROM python:3.11-slim

# Системные зависимости
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Код приложения
COPY . .

# Папка для FAISS
RUN mkdir -p faiss_index

EXPOSE 8000

CMD ["python", "bot.py"]