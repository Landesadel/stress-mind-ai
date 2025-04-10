import tensorflow as tf
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

os.makedirs('training/metrics', exist_ok=True)

# Загрузка данных
df = pd.read_csv('data/stress_data.csv')
X = df.drop('Mental Stress Level', axis=1)
y = df['Mental Stress Level']

# Нормализация и разделение
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Построение модели
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2(0.01, 0.01)),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mse', 'mae'])

# Колбэк для сохранения лучшей модели
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'models/stress_model.keras',
    save_best_only=True,
    monitor='val_loss',
    mode='min'
)

lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10
)

history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=64,
    validation_split=0.2,
    callbacks=[checkpoint, lr_scheduler],
    verbose=1
)

# Сохранение метрик
pd.DataFrame(history.history).to_csv('training/metrics/stress_metrics.csv', index=False)

# Настройка стиля графиков
sns.set_style("whitegrid")
plt.figure(figsize=(14, 6))

# Предсказания vs Реальные значения
plt.subplot(1, 2, 1)
sns.regplot(x=y_test, y=model.predict(X_test).flatten())
plt.plot([0,10], [0,10], 'r--')
plt.title('Actual vs Predicted Stress Levels')
plt.xlabel('Actual')
plt.ylabel('Predicted')

# График потерь
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'],
         linewidth=2,
         color='#2ca02c',
         label='Train Loss')
plt.plot(history.history['val_loss'],
         linewidth=2,
         color='#d62728',
         linestyle='--',
         label='Validation Loss')
plt.title('Потери модели', fontsize=14, pad=20)
plt.xlabel('Эпохи', fontsize=12)
plt.ylabel('Потери', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# Регулировка отступов и сохранение
plt.tight_layout()
plt.savefig('training/metrics/training_stress_model_history.png', dpi=300, bbox_inches='tight')
plt.close()
