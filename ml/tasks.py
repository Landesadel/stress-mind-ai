from invoke import task


@task
def data(c):
    """Запуск обработки и сохранения данных"""
    print("Подготовка данных...")
    c.run("python data/preprocessor.py")
    print("Данные подготовлены и сохранены:\ndata/stress_data.csv\ndata/mechanism_data.csv")
    print("Информацию о данных можно посмотреть в: data/metrics/")


@task
def train(c, model=None):
    """Запуск обучения моделей"""
    models = {
        'stress': 'training/train_stress_model.py',
        'mechanism': 'training/train_mechanisms.py'
    }

    if model:
        if model not in models:
            raise ValueError(f"Модель не найдена: {model}")
        print(f"Обучение {model} модели...")
        c.run(f"python {models[model]}")
    else:
        print("Обучение моделей...")
        for name, script in models.items():
            print(f"\nОбучение {name} модели:")
            c.run(f"python {script}")

    print("Информацию о результатах обучения можно посмотреть в: training/metrics")
    print("Сохраненные модели можно найти в: models/")


@task(pre=[data, train])
def start(c):
    """Запуск полного конвейера"""
    print("Конвейер успешно завершил работу")


@task
def clean(c):
    """Запуск очистки проекта от артефактов"""
    print("Очистка проекта...")
    c.run("rm -rf models/*")
    c.run("rm -rf models/metrics/*")
    c.run("rm -rf data/metrics/*")
    c.run("rm -f data/stress_data.csv")
    c.run("rm -f data/mechanism_data.csv")
    print("Проект очищен от артефактов!")
