import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

data_dir = '/home/igorralha/Documentos/dataset_cafe'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

print("--- Carregando dados de validação para avaliação ---")
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset='validation',
    seed=42,
    shuffle=False
)

class_names = val_ds.class_names

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
report = classification_report(y_true, y_pred, target_names=class_names)
print(report)

print("\n--- Gerando Matriz de Confusão ---")
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Blues', 
    xticklabels=class_names, 
    yticklabels=class_names
)
plt.xlabel('Previsão do Modelo', fontsize=12)
plt.ylabel('Classe Real (Ground Truth)', fontsize=12)
plt.title('Matriz de Confusão - Doenças do Café (DigiPathos)', fontsize=14)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()

output_png = 'matriz_confusao.png'
plt.savefig(output_png, dpi=300)
print(f"Matriz de confusão salva com sucesso em: {output_png}")