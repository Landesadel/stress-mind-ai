import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка и подготовка данных
df = pd.read_csv('data/mechanism_data.csv')

# Анализ распределения уровня стресса
plt.hist(df['Mental Stress Level'], bins=11)
plt.title('Распределение уровня стресса')
plt.show()

# Разделение данных
X = df[['Mental Stress Level']].values
y = df.drop('Mental Stress Level', axis=1).values

# Нормализация уровня стресса (0-10 -> 0-1)
X = X / 10.0

# Разделение на тренировочный и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Построение перцептрона с несколькими выходами
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(y.shape[1], activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
    loss='binary_crossentropy',
    metrics=[
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc')
    ]
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Обучение модели
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

pd.DataFrame(history.history).to_csv('training/metrics/mechanism_metrics.csv', index=False)

# Строим график
sns.set_style("whitegrid")
plt.figure(figsize=(14, 6))

# График auc
plt.subplot(1, 2, 1)
plt.plot(history.history['auc'],
         linewidth=2,
         color='#1f77b4',
         label='Train auc')
plt.plot(history.history['val_auc'],
         linewidth=2,
         color='#ff7f0e',
         linestyle='--',
         label='Validation auc')
plt.title('Способность модели разделять классы', fontsize=14, pad=20)
plt.xlabel('Эпохи', fontsize=12)
plt.ylabel('auc', fontsize=12)
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# Регулировка отступов и сохранение
plt.tight_layout()
plt.savefig('training/metrics/training_mechanisms_model_history.png', dpi=300, bbox_inches='tight')
plt.close()
