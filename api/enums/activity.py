from enum import Enum


class Activity(Enum):
    """Класс-перечисление для активностей с русскими и английскими метками."""
    NATURE_WALKS = "Прогулки на улице, или на природе"
    MEDITATION = "Медитативные практики"
    READING = "Чтение интересных вам книг"
    SOCIAL_EVENTS = "Посещение интересных вам общественных мероприятий"
    PHYSICAL_EXERCISE = "Физические упражнения (фитнес/плавание/бег и прочее)"
    FRIENDS_TIME = "Проведение времени в компании друзей/друга"
    YOGA = "Йога"
    MEDIA_CONSUMPTION = "Просмотр видео/фильмов/сериалов"
    SELF_CARE = "Посвятить время себе"
    NEW_PLACES = "Посещение новых мест"

    @classmethod
    def get_russian_names(cls, english_keys: list[str]) -> str:
        """
        Преобразует список английских ключей в строку с русскими названиями.

        :param english_keys: (list[str]) ключи-значения
        :return: str
        """
        result = []
        for key in english_keys:
            try:
                # Игнорируем регистр входных ключей
                activity = cls[key.upper()]
                result.append(activity.value)
            except KeyError:
                pass
        return ", ".join(result)
