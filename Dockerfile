FROM python:3.12-slim
LABEL authors="landesadel"

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Сначала копируем только зависимости
COPY pyproject.toml .

# Установка Python зависимостей
RUN pip install --no-cache-dir uv && \
    uv pip install --no-deps -r pyproject.toml

# Затем копируем весь код
COPY . .

EXPOSE 8000

# Запуск приложения
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]