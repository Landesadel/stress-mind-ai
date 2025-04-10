import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from dotenv import load_dotenv

load_dotenv()

dataset = os.getenv('DATASET')

df = pd.read_csv(f'../datasets/{dataset}')
os.makedirs('data/metrics', exist_ok=True)

print('Подготовка данных к анализу')

df.drop('User ID', axis=1, inplace=True)
df['Gender'] = df['Gender'].apply(lambda x: 1 if x == 'Male' else 0)
df['Counseling Attendance'] = df['Counseling Attendance'].apply(lambda x: 1 if x == 'Yes' else 0)
mechanisms = df['Stress Coping Mechanisms'].str.get_dummies(sep=',')
df = pd.concat([df.drop('Stress Coping Mechanisms', axis=1), mechanisms], axis=1)

print('Начало EDA анализа...')

# 1. Общая информация
print("\n[1] Первые 5 строк:")
print(df.head())

print("\n[2] Информация о данных:")
print(df.info())

# 2. Проверка пропусков
missing_values = df.isnull().sum()
print("\n[3] Пропущенные значения:")
print(missing_values)

# 3. Анализ целевой переменной
plt.figure(figsize=(8, 6))
sns.countplot(x='Mental Stress Level', data=df)
plt.title('Распределение целевой переменной')
plt.savefig('data/metrics/target_distribution.png')
plt.close()

print("\n[4] Распределение классов:")
print(df['Mental Stress Level'].value_counts(normalize=True))

# 4. Поиск дубликатов
duplicates = df.duplicated().sum()
print(f"\n[5] Найдено дубликатов: {duplicates}")

# Удаление дубликатов
if duplicates > 0:
    print("Удаляем дубликаты...")
    df = df.drop_duplicates()

# 5. Анализ выбросов
plt.figure(figsize=(12, 8))
df.boxplot()
plt.xticks(rotation=45)
plt.title('Распределение признаков')
plt.savefig('data/metrics/box_plots.png', bbox_inches='tight')
plt.close()

print("\n[6] Описательная статистика:")
print(df.describe().T)

# 6. Корреляционный анализ
plt.figure(figsize=(12, 10))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Матрица корреляций')
plt.savefig('data/metrics/correlation_matrix.png')
plt.close()

# Создание комплексных признаков
df['total_weekly_hours'] = df['Study Hours Per Week'] + df['Work Hours Per Week']
df['sleep_efficiency'] = df['Sleep Duration (Hours per night)'] / df['Social Media Usage (Hours per day)'].clip(1)

print('Разделение и сохранение данных')

# Для модели предсказания стресса
mechanism_columns = df.columns[14:24].tolist()
stress_df = df.drop(columns=mechanism_columns)
stress_df.to_csv('./data/stress_data.csv', index=False)

# Для модели рекомендаций
mechanism_columns = df.columns[14:24].tolist()
mechanism_df = df[['Mental Stress Level'] + mechanism_columns]
mechanism_df.to_csv('./data/mechanism_data.csv', index=False)
