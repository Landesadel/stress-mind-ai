import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Загрузка данных
df = pd.read_csv('data/mechanism_data.csv')
X = df[['Mental Stress Level']]
y = df.drop('Mental Stress Level', axis=1)

# Преобразование данных
X = X.values.reshape(-1, 1)
y = y.values

# Упрощенная модель
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(X.shape[1],)),
    tf.keras.layers.Dense(y.shape[1], activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', 'precision', 'recall', 'auc']
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'models/mechanism_model.keras',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_auc',
    patience=20,
    mode='max',
    restore_best_weights=True
)

history = model.fit(
    X, y,
    epochs=100,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stop, checkpoint],
    verbose=1
)

pd.DataFrame(history.history).to_csv('training/metrics/mechanism_metrics.csv', index=False)

# Строим график
sns.set_style("whitegrid")
plt.figure(figsize=(14, 6))

# График точности
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'],
         linewidth=2,
         color='#1f77b4',
         label='Train Accuracy')
plt.plot(history.history['val_accuracy'],
         linewidth=2,
         color='#ff7f0e',
         linestyle='--',
         label='Validation Accuracy')
plt.title('Точность модели', fontsize=14, pad=20)
plt.xlabel('Эпохи', fontsize=12)
plt.ylabel('Точность', fontsize=12)
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# Регулировка отступов и сохранение
plt.tight_layout()
plt.savefig('training/etrics/training_mechanisms_model_history.png', dpi=300, bbox_inches='tight')
plt.close()
