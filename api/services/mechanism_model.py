import numpy as np
import tensorflow as tf
import pandas as pd


class MechanismModel:
    def __init__(self) -> None:
        self.mechanism_data = pd.read_csv('./ml/data/mechanism_data.csv')
        self.model = tf.keras.models.load_model('./ml/models/mechanisms_model.keras')

    def predict(self, data: int) -> list:
        """
        На основании уровня стресса, выдает список рекомендаций по механизмам его преодоления.

        :param data: (int) Целочисленное значение - уровень стресса
        :return: list: Список рекомендованных механизмов
        """
        stress_level = 10 if data >= 10 else data
        stress_level = np.array([stress_level])
        mechanisms = self.model.predict([stress_level])[0]

        return [self.mechanism_data.columns[i + 1] for i, val in enumerate(mechanisms) if val > 0.5]
