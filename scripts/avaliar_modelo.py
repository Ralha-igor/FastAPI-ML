import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

data_dir = '/home/igorralha/Documentos/dataset_cafe'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

print("--- Carregando dados de validação para avaliação ---")
# Mude de shuffle=False para shuffle=True:
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset='validation',
    seed=42,
    shuffle=True  # <--- Habilite o shuffle com a mesma seed de treino!
)

class_names = val_ds.class_names
num_classes = len(class_names)
labels_index = list(range(num_classes))

print("\n--- Carregando modelo 'modelo_cafe.keras' ---")
model = tf.keras.models.load_model('modelo_cafe.keras', compile=False)

print("\n--- Gerando previsões ---")
y_true = []
y_pred = []

for images, labels in val_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

print("\n========================================================")
print("             RELATÓRIO DE DESEMPENHO (VAL)             ")
print("========================================================")
# Passamos labels=labels_index para garantir as 6 classes mesmo se alguma faltar no val_ds
report = classification_report(y_true, y_pred, labels=labels_index, target_names=class_names, zero_division=0)
print(report)

print("\n--- Gerando Matriz de Confusão ---")
cm = confusion_matrix(y_true, y_pred, labels=labels_index)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Blues', 
    xticklabels=class_names, 
    yticklabels=class_names
)
plt.xlabel('Previsão do Modelo')
plt.ylabel('Classe Real')
plt.title('Matriz de Confusão - Doenças do Café')
plt.tight_layout()
plt.savefig('matriz_confusao.png')
print("Gráfico da Matriz de Confusão salvo em 'matriz_confusao.png'!")
