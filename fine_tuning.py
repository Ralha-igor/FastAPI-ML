import tensorflow as tf
from tensorflow.keras import layers, models

data_dir = '/home/igorralha/Documentos/dataset_cafe'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

print("--- Carregando dados para Fine-Tuning ---")
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset='training',
    seed=42
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset='validation',
    seed=42
)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# 1. Carregar o modelo treinado anteriormente
print("\n--- Carregando modelo base treinado ---")
model = tf.keras.models.load_model('modelo_cafe.keras')

# 2. Localizar a camada base (MobileNetV2) e descongelar as últimas camadas
base_model = None
for layer in model.layers:
    if 'mobilenet' in layer.name.lower():
        base_model = layer
        break

if base_model:
    base_model.trainable = True
    # Mantém congeladas as primeiras camadas e descongelamos apenas as últimas 30
    fine_tune_at = len(base_model.layers) - 30
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    print(f"Camadas descongeladas a partir da camada {fine_tune_at} do MobileNetV2.")
else:
    print("Aviso: Modelo base não identificado diretamente, descongelando camadas superiores.")

# 3. Recompilar com Taxa de Aprendizado (Learning Rate) bem baixa
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 4. Executar Fine-Tuning (5 a 10 épocas)
print("\n--- Iniciando etapa de Fine-Tuning ---")
history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=8
)

# 5. Salvar o modelo otimizado
model.save('modelo_cafe_finetuned.keras')
print("\nFine-Tuning concluído com sucesso! Modelo salvo em 'modelo_cafe_finetuned.keras'.")
