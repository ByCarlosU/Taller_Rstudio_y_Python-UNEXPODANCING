import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import plot_tree

# Cargar dataset
data = pd.read_csv("dance_dataset.csv")

# Separar variables
X = data[["left_arm", "right_arm", "left_leg", "right_leg"]]
y = data["label"]

# Dividir en entrenamiento y testing (80% - 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Crear modelo Árbol de Decisión
model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=4,
    random_state=42
)

# Entrenar modelo
model.fit(X_train, y_train)

# Predicción
y_pred = model.predict(X_test)

# Evaluación
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy del modelo:", accuracy)
print("\nMatriz de Confusión:\n")
print(confusion_matrix(y_test, y_pred))
print("\nReporte de Clasificación:\n")
print(classification_report(y_test, y_pred))

# Gráfico del árbol
plt.figure(figsize=(18,10))
plot_tree(
    model,
    feature_names=["left_arm", "right_arm", "left_leg", "right_leg"],
    class_names=model.classes_,
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("Árbol de Decisión - UnexpoDancing")

plt.show()
