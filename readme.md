# ☕ Classificador de Doenças do Cafeeiro

API de **classificação de doenças em folhas de cafeeiro**, desenvolvida com **Deep Learning, Transfer Learning, TensorFlow/Keras e FastAPI**.

O projeto utiliza a arquitetura **MobileNetV2**, pré-treinada com ImageNet, para classificar imagens em seis categorias de doenças do cafeeiro. O modelo é disponibilizado através de uma API REST e está hospedado na plataforma **Render**.

> **Status:** 🚀 API disponível em produção

---

## 📌 Sobre o projeto

O objetivo do projeto é disponibilizar um serviço capaz de receber uma imagem de uma folha de café e retornar a doença que apresenta a maior probabilidade de ocorrência.

A pipeline completa segue o fluxo:

```text
Dataset DigiPathos
       ↓
Pré-processamento
       ↓
Data Augmentation
       ↓
Transfer Learning
       ↓
MobileNetV2
       ↓
Modelo Keras (.keras)
       ↓
FastAPI
       ↓
Render
       ↓
API REST de Inferência
```

O dataset utilizado é o **DigiPathos**, disponibilizado pela Embrapa.

**Fonte:** DigiPathos — Embrapa
**DOI:** `10.48432/XA1OVL`

---

## 🧠 Modelo

O modelo foi desenvolvido utilizando **Transfer Learning** com a arquitetura **MobileNetV2**.

### Configuração

* **Arquitetura:** MobileNetV2
* **Pesos iniciais:** ImageNet
* **Entrada:** `224 × 224 × 3`
* **Classes:** 6
* **Otimizador:** Adam
* **Loss:** `categorical_crossentropy`
* **Classificador:** Dense com `softmax`
* **Dropout:** utilizado no cabeçote de classificação
* **Formato do modelo:** `.keras`

As camadas convolucionais da MobileNetV2 foram congeladas durante o treinamento, utilizando a rede pré-treinada como extratora de características.

---

## 🏷️ Classes

O modelo trabalha com seis classes:

| Índice | Classe             |
| -----: | ------------------ |
|      0 | `bicho_mineiro`    |
|      1 | `cercosporiose`    |
|      2 | `ferrugem`         |
|      3 | `mancha_aureolada` |
|      4 | `mancha_mantegosa` |
|      5 | `phoma`            |

⚠️ **Importante:** o modelo **não possui uma classe "saudável"**.

Isso significa que qualquer imagem enviada recebe uma das seis classes, mesmo que seja uma folha saudável ou uma imagem que não seja de café.

---

# ⚙️ Pipeline de dados

## 1. Dataset

O dataset DigiPathos foi dividido em:

* **80%:** treinamento
* **20%:** validação
* **Seed:** `42`

A divisão foi realizada de forma que os subconjuntos não se sobreponham.

---

## 2. Pré-processamento

As imagens são padronizadas para:

```text
224 × 224 × 3
```

com lotes de:

```text
batch_size = 32
```

O modelo utiliza normalização dos pixels para o intervalo:

```text
[-1, 1]
```

Essa normalização está incorporada ao próprio modelo antes da MobileNetV2.

Por isso, a API recebe imagens com pixels no intervalo original `0–255` e **não realiza uma segunda normalização**.

---

## 3. Data Augmentation

Durante o treinamento são aplicadas técnicas de aumento de dados:

* `RandomFlip` horizontal;
* `RandomFlip` vertical;
* `RandomRotation`;
* `RandomZoom`.

O augmentation é aplicado somente aos dados de treinamento.

---

# 📂 Estrutura do projeto

Uma estrutura esperada para o repositório é:

```text
FastAPI-ML/
│
├── main.py
├── modelo_cafe.keras
├── requirements.txt
├── .python-version
│
├── script_de_avalia_o_de_m_tricas_e_matriz_de_confus_o.py
├── figuras.zip
│
└── README.md
```

### Principais arquivos

| Arquivo                                                  | Função                               |
| -------------------------------------------------------- | ------------------------------------ |
| `main.py`                                                | API FastAPI e inferência do modelo   |
| `modelo_cafe.keras`                                      | Modelo treinado                      |
| `requirements.txt`                                       | Dependências do projeto              |
| `.python-version`                                        | Versão do Python utilizada no deploy |
| `script_de_avalia_o_de_m_tricas_e_matriz_de_confus_o.py` | Avaliação do modelo                  |
| `figuras.zip`                                            | Figuras e resultados da avaliação    |

---

# 🚀 Como executar localmente

## 1. Clonar o repositório

```bash
git clone https://github.com/Ralha-igor/FastAPI-ML.git
```

Entrar na pasta:

```bash
cd FastAPI-ML
```

---

## 2. Criar ambiente virtual

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Instalar as dependências

```bash
pip install -r requirements.txt
```

As versões utilizadas no ambiente de produção são:

```text
fastapi==0.110.0
uvicorn==0.28.0
tensorflow-cpu==2.18.0
keras==3.15.1
pillow==10.2.0
python-multipart==0.0.9
numpy==1.26.4
```

---

## 4. Iniciar a API

Execute:

```bash
uvicorn main:app --reload
```

A API ficará disponível localmente em:

```text
http://127.0.0.1:8000
```

---

# 🧪 Testando a API

O FastAPI disponibiliza automaticamente uma interface Swagger.

Abra:

```text
http://127.0.0.1:8000/docs
```

Nela é possível testar o endpoint `/predict` diretamente pelo navegador.

---

# 🔌 Endpoints

## `GET /`

Utilizado para verificar se a API está funcionando.

### Resposta

```json
{
  "status": "online",
  "modelo": "MobileNetV2 DigiPathos"
}
```

---

## `POST /predict`

Recebe uma imagem e realiza a classificação.

A imagem deve ser enviada no campo:

```text
file
```

São aceitos:

```text
JPEG
PNG
```

### Exemplo utilizando cURL

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "accept: application/json" \
  -F "file=@/caminho/para/folha_cafe.jpg;type=image/jpeg"
```

---

# 📤 Exemplo de resposta

A API retorna o diagnóstico, a confiança da previsão e a distribuição de probabilidades entre as seis classes.

```json
{
  "diagnostico": "phoma",
  "confianca": 88.73,
  "probabilidades": {
    "bicho_mineiro": 1.34,
    "cercosporiose": 8.23,
    "ferrugem": 0.20,
    "mancha_aureolada": 1.49,
    "mancha_mantegosa": 0.01,
    "phoma": 88.73
  }
}
```

As probabilidades totalizam aproximadamente 100%.

---

# ☁️ API em produção

A API também foi publicada no **Render**.

### API

https://fastapi-ml-7fcy.onrender.com

### Swagger

https://fastapi-ml-7fcy.onrender.com/docs

### Endpoint de previsão

```text
POST https://fastapi-ml-7fcy.onrender.com/predict
```

---

# 🧪 Teste realizado em produção

Após o deploy, foi realizado um teste utilizando a imagem `phoma3.jpeg`.

A API retornou:

| Classe           | Probabilidade |
| ---------------- | ------------: |
| **phoma**        |    **88,73%** |
| cercosporiose    |         8,23% |
| mancha_aureolada |         1,49% |
| bicho_mineiro    |         1,34% |
| ferrugem         |         0,20% |
| mancha_mantegosa |         0,01% |

O diagnóstico retornado foi:

```text
phoma
```

Esse teste confirmou o funcionamento da pipeline de inferência, desde o carregamento do modelo até o formato da resposta da API.

> ⚠️ Esse teste é um **teste de fumaça (smoke test)**. Ele não deve ser interpretado como uma avaliação da qualidade geral ou da acurácia do modelo.

---

# 🛠️ Deploy

O deploy foi realizado utilizando o **Render** como Web Service.

### Configuração

```text
Runtime:
Python 3.11.9
```

### Build Command

```bash
pip install -r requirements.txt
```

### Start Command

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

A versão do Python é definida pelo:

```text
.python-version
```

e pela variável de ambiente:

```text
PYTHON_VERSION=3.11.9
```

⚠️ Os dois valores devem permanecer iguais.

---

# 🐛 Problemas encontrados durante o desenvolvimento

Durante o deploy foram encontrados alguns problemas de compatibilidade e configuração.

### 1. Incompatibilidade do Keras

O modelo havia sido salvo utilizando:

```text
Keras 3.15.1
```

Enquanto o ambiente inicial utilizava uma versão incompatível.

Isso causava o erro:

```text
GlorotUniform.__init__() got an unexpected keyword argument 'input_axes'
```

### Solução

O ambiente foi ajustado para:

```text
Python 3.11.9
TensorFlow 2.18.0
Keras 3.15.1
```

Após limpar o cache do build e realizar um novo deploy, o modelo foi carregado corretamente.

---

### 2. Erro na versão do Python

O Render estava tentando utilizar:

```text
Python 3.11.09
```

O valor estava incorreto devido a um zero adicional.

### Solução

A variável foi corrigida para:

```text
PYTHON_VERSION=3.11.9
```

---

### 3. Erro de autenticação do GitHub

Durante o envio do projeto para o GitHub foi encontrado:

```text
Password authentication is not supported
```

O GitHub não permite mais autenticação por senha para operações Git via HTTPS.

A solução foi utilizar um **Personal Access Token (PAT)**.

---

### 4. Bloqueio do navegador

O frontend que consumia a API apresentou bloqueio devido à política de origem do navegador.

A solução foi adicionar o middleware:

```python
CORSMiddleware
```

com suporte às requisições necessárias.

> Para produção, o ideal é substituir `allow_origins=["*"]` pelo domínio específico da aplicação.

---

# ⚠️ Limitações

Este projeto possui algumas limitações importantes.

### Sem classe saudável

O modelo sempre retorna uma das seis doenças.

Uma possível evolução seria adicionar uma classe:

```text
saudável
```

ou implementar um mecanismo de rejeição para previsões com baixa confiança.

### Confiança não significa certeza

O valor retornado pela API é derivado da distribuição `softmax`.

Uma confiança elevada não garante que o diagnóstico esteja correto.

O resultado deve ser tratado como uma **hipótese computacional**, e não como diagnóstico agronômico definitivo.

### Imagens fora do domínio

Imagens que não sejam de folhas de café também podem receber uma classificação.

### Render

A versão gratuita do Render pode hibernar após períodos de inatividade.

Consequentemente, a primeira requisição após a hibernação pode apresentar maior tempo de resposta.

---

# 🔐 Privacidade

A API processa a imagem em memória durante a requisição.

A implementação não grava a imagem em disco nem em logs, e o redimensionamento para `224 × 224` descarta os metadados EXIF, incluindo informações de GPS.

Ainda assim, como a inferência é realizada remotamente, a aplicação que envia a imagem deve informar ao usuário que a fotografia é enviada para análise.

---

# 📚 Tecnologias utilizadas

* Python
* TensorFlow
* Keras
* MobileNetV2
* FastAPI
* Uvicorn
* Pillow
* NumPy
* Render
* Git
* GitHub

---

# 📊 Avaliação do modelo

A avaliação completa do modelo é realizada através do script:

```text
script_de_avalia_o_de_m_tricas_e_matriz_de_confus_o.py
```

Os resultados incluem:

* acurácia de validação;
* matriz de confusão;
* métricas por classe.

Os resultados completos estão disponíveis no repositório através do arquivo:

```text
figuras.zip
```

O teste realizado diretamente na API não substitui essa avaliação.

---

# 📖 Referências

### Dataset

**DigiPathos — Embrapa**

DOI:

```text
10.48432/XA1OVL
```

### Repositório

**GitHub — Ralha-igor/FastAPI-ML**

```text
https://github.com/Ralha-igor/FastAPI-ML
```

---

## 👨‍💻 Autor

**Igor Ralha**

Projeto desenvolvido como aplicação prática de **Machine Learning, Deep Learning, Engenharia de Dados e desenvolvimento de APIs**, integrando treinamento de modelo, inferência e disponibilização em ambiente de produção.

---

## ⚠️ Aviso

Este projeto possui finalidade **educacional e experimental**.

As previsões produzidas pelo modelo não devem ser utilizadas isoladamente para tomada de decisão agronômica. Para diagnóstico definitivo, recomenda-se avaliação por profissional qualificado.

