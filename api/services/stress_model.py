import tensorflow as tf
import numpy as np
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
        df = data.copy()
        df['total_weekly_hours'] = df['Study Hours Per Week'] + df['Work Hours Per Week']
        df['sleep_efficiency'] = df['Sleep Duration (Hours per night)'] / df['Social Media Usage (Hours per day)'].clip(
            1)

        predict = self.model.predict(df)[0][0]
        stress_level = np.clip(predict, 1, 10).astype(int)
        return stress_level
