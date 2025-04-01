import os
import uuid
import logging
import requests
from dotenv import load_dotenv

load_dotenv()


class GigaChatModel:
    def __init__(self):
        self.auth_token = os.getenv('GIGA_AUTH')
        self.scope = os.getenv('GIGA_SCOPE')
        self.token_url = os.getenv('TOKEN_URL')
        self.chat_url = os.getenv('CHAT_URL')

        # Настройка логирования
        logging.basicConfig(level=logging.ERROR)
        self.logger = logging.getLogger(__name__)

    def get_token(self) -> str:
        """
        Получение токена доступа для API GigaChat

        :return: str  Access token
        """
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Accept': 'application/json',
            'RqUID': str(uuid.uuid4()),
            'Authorization': f'Basic {self.auth_token}'
        }

        payload = {
            'scope': self.scope
        }

        try:
            response = requests.post(
                self.token_url,
                headers=headers,
                data=payload,
                verify=False
            )

            if response.status_code == 200:
                return response.json().get('access_token')
            else:
                self.logger.error(f"Token request failed: {response.status_code} - {response.text}")
                return ''

        except Exception as e:
            self.logger.error(f"Error getting token: {str(e)}")
            return ''

    def get_answer(self, message, token) -> str:
        """
        Получение ответа от модели GigaChat

        :param message:
        :param token: ключ авторизации
        :return: str: ответ модели
        """
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'Authorization': f'Bearer {token}'
        }

        content = [
            {
                'role': 'system',
                'content': 'Отвечай как ассистент-помощник. Коротко и понятно.'
            },
            {
                'role': 'user',
                'content': message
            }
        ]

        payload = {
            "model": "GigaChat",
            "messages": content,
            "temperature": 0.8,
            "top_p": 0.1,
            "n": 1,
            "stream": False,
            "max_tokens": 220,
            "repetition_penalty": 1,
            "update_interval": 0
        }

        try:
            response = requests.post(
                self.chat_url,
                headers=headers,
                json=payload,
                verify=False
            )

            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            else:
                self.logger.error(f"Chat request failed: {response.status_code} - {response.text}")
                return ""

        except Exception as e:
            self.logger.error(f"Error getting answer: {str(e)}")
            return ""
