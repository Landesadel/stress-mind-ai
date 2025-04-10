FROM python:3.12-slim
LABEL authors="landesadel"

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Копируем весь код
COPY . .

# Установка Python зависимостей
RUN pip install uv
RUN uv sync

EXPOSE 8000

# Запуск приложения
CMD [".venv/bin/python",  "-m",  "api.main"]