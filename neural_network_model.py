import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Cargar dataset
data = pd.read_csv("dance_dataset.csv")

X = data[["left_arm", "right_arm", "left_leg", "right_leg"]]
y = data["label"]

# Convertir etiquetas a números
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Escalar datos (MUY importante en redes neuronales)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Separar entrenamiento y testing
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42
)

# Crear modelo
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(12, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Entrenar modelo
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)

# Evaluar modelo
loss, accuracy = model.evaluate(X_test, y_test)
print("\nAccuracy Red Neuronal:", accuracy)

# Predicciones
y_pred = np.argmax(model.predict(X_test), axis=1)

print("\nMatriz de Confusión:")
print(confusion_matrix(y_test, y_pred))

print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

# Gráfica de entrenamiento
plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title("Accuracy Red Neuronal")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()