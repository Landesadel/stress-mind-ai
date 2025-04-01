import tensorflow as tf
import math
from pandas import DataFrame


class StressModel:
    def __init__(self) -> None:
        self.model = tf.keras.models.load_model('../../ml/models/stress_model.keras')

    def predict(self, data: DataFrame) -> int:
        """
        Выполняет предсказание с помощью обученной модели и возвращает целочисленный результат.

        :param data: (dict) Входные данные в формате словаря, соответствующие требованиям модели
        :return: int: Прогнозируемое значение
        """

        predict = self.model.predict(data)[0][0]
        stress_level = math.ceil(abs(predict))
        return stress_level
