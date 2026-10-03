import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array

# 1. Mapeamento das 6 classes na ordem alfabética que o Keras organizou
CLASSES = [
    'bicho_mineiro',
    'cercosporiose',
    'ferrugem',
    'mancha_aureolada',
    'mancha_mantegosa',
    'phoma'
]

def classificar_imagem(caminho_imagem, caminho_modelo='modelo_cafe.keras'):
    # Carregar modelo
    print(f"--- Carregando modelo: {caminho_modelo} ---")
    model = tf.keras.models.load_model(caminho_modelo, compile=False)

    # Carregar e redimensionar a imagem para 224x224
    img = load_img(caminho_imagem, target_size=(224, 224))
    
    # Converter para array e criar a dimensão de batch (1, 224, 224, 3)
    img_array = img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)

    # Executar inferência
    predicoes = model.predict(img_batch, verbose=0)[0]
    
    # Identificar a classe com maior probabilidade
    idx_classe = np.argmax(predicoes)
    classe_prevista = CLASSES[idx_classe]
    confianca = predicoes[idx_classe] * 100

    print("\n========================================================")
    print(f"Imagem: {caminho_imagem}")
    print(f"Diagnóstico: {classe_prevista.upper()}")
    print(f"Confiança:   {confianca:.2f}%")
    print("========================================================")
    
    # Exibir a distribuição completa das probabilidades
    print("\nProbabilidade por classe:")
    for i, nome_classe in enumerate(CLASSES):
        print(f" - {nome_classe:<18}: {predicoes[i]*100:6.2f}%")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python3 predicao.py /caminho/da/imagem.jpg")
    else:
        caminho_img = sys.argv[1]
        classificar_imagem(caminho_img)
