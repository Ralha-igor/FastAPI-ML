import tensorflow as tf
import numpy as np
import os

# 1. Carregar o modelo no formato novo
model = tf.keras.models.load_model('modelo_cafe.keras', compile=False)

# Mapeamento das classes
class_names = ['bicho_mineiro', 'cercosporiose', 'ferrugem', 'mancha_aureolada', 'mancha_mantegosa', 'phoma']

def prever_pasta(caminho_pasta):
    if not os.path.exists(caminho_pasta):
        print(f"Erro: A pasta '{caminho_pasta}' não foi encontrada.")
        return

    arquivos = [f for f in os.listdir(caminho_pasta) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not arquivos:
        print("Nenhuma imagem (.jpg, .jpeg, .png) foi encontrada na pasta.")
        return

    print(f"\n--- Processando {len(arquivos)} imagem(ns) da pasta: {caminho_pasta} ---\n")

    for arquivo in sorted(arquivos):
        caminho_imagem = os.path.join(caminho_pasta, arquivo)

        # Prepara a imagem com as transformações da MobileNet
        img = tf.keras.utils.load_img(caminho_imagem, target_size=(224, 224))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        # Previsão
        predictions = model.predict(img_array, verbose=0)
        classe_prevista = class_names[np.argmax(predictions[0])]
        confianca = 100 * np.max(predictions[0])

        print(f"📷 Imagem: {arquivo}")
        print(f"   Diagnóstico: {classe_prevista.upper()} | Confiança: {confianca:.2f}%\n")

pasta_figuras = '/home/igorralha/Documentos/figuras'
prever_pasta(pasta_figuras)
