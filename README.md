# StressMind AI: Прогнозирование уровня стресса с рекомендациями 🧠✨

**StressMind AI** — это интеллектуальный помощник для прогнозирования уровня стресса на основе поведенческих и социальных факторов. Проект предоставляет персонализированные рекомендации для снижения стресса, используя машинное обучение и современный API на базе FastAPI.

[FastAPI](https://fastapi.tiangolo.com/ru/learn/)
[Docker](https://docs.docker.com/manuals/)
[Python 3.12](https://docs.python.org/3.12/)

## ✨ Особенности

- **Прогнозирование стресса**: Нейронные сети в связке с Моделью LLM анализирует 13 параметров (возраст, сон, нагрузка и др.).
- **Рекомендации**: Персонализированные советы на основе прогноза.
- **FastAPI**: Высокопроизводительный API с автоматической документацией (Swagger/Redoc).
- **Docker-контейнер**: Простое развертывание и масштабирование.
- **Метрики обучения**: Отслеживание качества моделей в реальном времени.

## 🧠 Модели машинного обучения

Подробное описание архитектуры нейронных сетей, процесса обучения и технических решений доступно в отдельном документе:  
[📚 Системные детали](MODEL_DETAILS.md)

## 🚀 Быстрый старт

### Установка и настройка

#### 1. **Клонируйте репозиторий**:
```bash
git clone https://github.com/Landesadel/stress-mind-ai.git
cd stress-mind-ai
```
   
#### 2. Создайте виртуальное окружение (используем uv):

```bash
pip install uv

# Выполняется в корне проекта
# Устанавливает все указанные зависимости в pyproject.toml и uv.lock
uv sync
```

> Дальнейший запуск можно делать через активацию окружения `source venv/bin/activate1`  # Linux/MacOS  для Windows: `.\venv\Scripts\activate`
> 
> Можно через запуск самих скриптов `uv run <скрипт>` Важно помнить, что `run` использует системный пайтон, и
>конкретную версию можно указать параметром `--python=3.12`

#### 3. Подготовьте данные:

- [Скачайте датасет](https://drive.google.com/file/d/1JtBylHOl705pl7nf0i2cAc-dLcAytvz_/view?usp=sharing).

- Поместите файл в созданную вами папку datasets/ корневой директории проекта

- Обучите модель:

```bash
cd ml
invoke start  # Запуск пайплайна обработки данных и обучения
```
- Результаты обучения: `ml/training/metrics/`
- Данные о датасете: `ml/data/metrics`
- Модели сохранены в `ml/models/`

#### 4. Сборка в docker-контейнер
Запуск в Docker

```bash
docker build -t stress-mind-ai .
docker run -p 8000:8000 --env-file .env stress-mind-ai
```

---

### 📡 Использование API

Пример запроса cURL:

```bash
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '{
    "age": 25,
    "gender": 1,
    "study_hours_per_week": 15,
    "social_media_usage": 20,
    "sleep_duration": 8,
    "physical_exercise": 2,
    "family_support": 3,
    "financial_stress": 4,
    "peer_pressure": 4,
    "relationship_stress": 2,
    "counseling_attendance": 1,
    "food_quality": 3,
    "work_hours_per_week": 45
}'
```

Python:

```python
import requests

data = {
    "age": 25,
    "gender": 1,
    "study_hours_per_week": 15,
    "social_media_usage": 20,
    "sleep_duration": 8,
    "physical_exercise": 2,
    "family_support": 3,
    "financial_stress": 4,
    "peer_pressure": 4,
    "relationship_stress": 2,
    "counseling_attendance": 1,
    "food_quality": 3,
    "work_hours_per_week": 40
}

response = requests.post("http://localhost:8000/predict", json=data)
print(response.json())
```

Пример ответа
```json
{
  "advice": "Ты молодец, что обратил внимание на свое состояние! 🌟\n1. Сделай 5 глубоких вдохов: вдох на 4 счета, выдох на 6.\n2. Попробуй короткую медитацию: закрой глаза и сосредоточься на звуках вокруг.\n3. Выпей воды — это поможет успокоиться.\nТы справишься, я верю в тебя! 💪"
}
```

---

### 📚 Документация API
После запуска сервера откройте:

Swagger UI: http://localhost:8000/docs

ReDoc: http://localhost:8000/redoc

---


### 📄 Лицензия
MIT License. Подробнее в файле LICENSE.

---

💡 **Совет**: Перед обучением модели убедитесь, что датасет загружен в `datasets/`. 
