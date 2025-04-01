import math

import requests
from fastapi import APIRouter
import pandas as pd
import asyncio
from ..enums.activity import Activity
from ..schemas.user_data import UserData
from ..schemas.response import AdviceResponse
from ..services.gigachat_model import GigaChatModel
from ..services.mechanism_model import MechanismModel
from ..services.stress_model import StressModel
import os
from dotenv import load_dotenv

load_dotenv()

model_url = os.getenv('MODEL_URL')
model_token = os.getenv('MODEL_TOKEN')
router = APIRouter()


@router.post("/predict", response_model=AdviceResponse)
async def process_predict(data: UserData):
    """

    :param data: данные для предсказаний
    :return: str ответ модели
    """
    df_data = pd.DataFrame([data])

    stress_model = StressModel()
    stress_level = await asyncio.create_task(stress_model.predict(df_data))
    stress_level = math.ceil(abs(stress_level)[0][0])

    mechanism_model = MechanismModel()
    recommend_mechanisms = mechanism_model.predict(stress_level)
    activities = Activity.get_russian_names(recommend_mechanisms)

    # Генерация ответа
    prompt = f"""
[Роль]
«Ты — эмпатичный бот-психолог, который помогает пользователям справляться со стрессом. Твоя задача:

[Алгоритм действий]
1. Оцени уровень стресса (от 1 до 10):
    1-3 (низкий): Подчеркни, что это нормально, похвали за осознанность.
    4-7 (средний): Вырази поддержку, предложи конкретные действия.
    8-10 (высокий): Сосредоточься на экстренных методах успокоения, подбодри.

2. Учти рекомендации по деятельности, если они есть:
Интегрируй их в советы (напр., "Попробуй прогулку, которую ты упомянул: это снизит напряжение").
Если рекомендации по деятельности отсутствуют — предложи свои (дыхательные техники, физическая активность, хобби, медитация).

[Структура ответа]
1.Начни с эмпатии: "Я понимаю, как тебе тяжело…" / "Ты молодец, что заботишься о себе!"
2. Дай 2-3 кратких совета, соответствующих уровню стресса.
3. Если есть рекомендации по деятельности, добавь их в ответ.
3. Добавь ободряющую фразу в конце (напр., "Ты справишься! Я верю в тебя 🌟").

[Стиль]
Дружелюбный, без сложных терминов.
Используй эмодзи для эмоциональной поддержки (🌿, 💪, 🌈).
Избегай шаблонов, стремись к персонализации.

[Входные данные]
Уровень стресса: {stress_level}
Рекомендованная активность: {activities}
"""

    llm = GigaChatModel()
    token = llm.get_token()
    answer = llm.get_answer(prompt, token)

    return {"advice": answer}
