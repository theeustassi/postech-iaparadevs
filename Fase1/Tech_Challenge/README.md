# 🏥 Sistema de Apoio ao Diagnóstico de Câncer de Mama

## Tech Challenge - Fase 1 | FIAP Pós-Tech IA para Devs

Olá! Bem-vindo ao nosso projeto :)

Desenvolvemos este sistema como parte do Tech Challenge da FIAP. A ideia aqui é usar Machine Learning para ajudar no diagnóstico de câncer de mama - classificando tumores como benignos ou malignos com base em características extraídas de exames de biópsia.

Basicamente treinamos alguns modelos para analisar dados de células e tentar prever se um tumor é perigoso ou não. É tipo ter um assistente inteligente que olha os dados e dá uma opinião baseada em padrões que aprendeu de casos anteriores.

## 🎯 O que fizemos aqui?

Nosso sistema:
- Analisa 30 características diferentes das células (tamanho, textura, formato, etc.)
- Testa 5 algoritmos diferentes de ML para ver qual funciona melhor
- Explica as decisões usando SHAP (pra gente entender o "porquê" por trás das previsões)
- Gera gráficos e visualizações bem legais dos resultados

> ⚕️ **Obs importante**: Isso é um trabalho acadêmico! Na vida real, a decisão final sempre tem que ser de um médico de verdade. Nosso modelo é só uma ferramenta de apoio mesmo.

### Estrutura do Projeto
```
Tech/
├── data/
│   └── breast_cancer_data.csv
├── notebooks/
│   └── diagnostico_cancer_mama.ipynb
├── src/
│   ├── preprocessing.py
│   ├── models.py
│   └── evaluation.py
├── results/
│   └── graficos/
├── Dockerfile
├── requirements.txt
└── README.md
```

### Pré-requisitos
- Python 3.9+
- Docker (opcional)

### Instalação

#### Opção 1: Instalação Local

**Windows (PowerShell)**:
```powershell
# Clone o repositório
git clone https://github.com/theeustassi/postech-iaparadevs.git
cd Tech

# Crie um ambiente virtual
python -m venv venv

# Ative o ambiente (PowerShell) - Execute uma das opções abaixo:
# Opção A: Permitir execução de scripts (recomendado)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Depois ative o ambiente:
.\venv\Scripts\Activate.ps1

# Ou Opção B: Ativar diretamente sem mudar política
& .\venv\Scripts\Activate.ps1

# Instale as dependências
pip install -r requirements.txt
```

**Linux/Mac**:
```bash
# Clone o repositório
git clone https://github.com/theeustassi/postech-iaparadevs.git
cd Tech

# Crie um ambiente virtual
python -m venv venv
source venv/bin/activate

# Instale as dependências
pip install -r requirements.txt
```

#### Opção 2: Docker
```bash
# Build da imagem
docker build -t diagnostico-cancer .

# Execute o container
docker run -p 8888:8888 diagnostico-cancer
```

### Como Executar

#### Notebook Jupyter
```bash
jupyter notebook notebooks/diagnostico_cancer_mama.ipynb
```

#### Scripts Python
```bash
python src/main.py
```

### 📊 Sobre os Dados

Usamos o **Wisconsin Breast Cancer Dataset** - um dataset bem conhecido na área:
- **Fonte**: [Kaggle/UCI](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
- **O que esse dataset contem?**: 569 casos com 30 características numéricas de cada tumor
- **Características**: Coisas como raio médio, textura, perímetro, área, suavidade... tudo extraído de imagens de biópsias
- **Objetivo**: Classificar se é maligno (M) ou benigno (B)

### 🛠️ Ferramentas que Usamos

Trabalhamos com Python e várias bibliotecas:
- **Pandas** e **NumPy** → para manipular os dados
- **Scikit-learn** → para os algoritmos de ML
- **Matplotlib/Seaborn** → gerar os gráficos
- **SHAP** → entender o que o modelo está "pensando"
- **Jupyter Notebook** → onde fizemos tudo de forma interativa

### 🤖 Modelos que Testamos

Rodamos 5 algoritmos diferentes pra ver qual seria melhor:
1. Regressão Logística (o mais simples)
2. Árvore de Decisão
3. Random Forest
4. K-Nearest Neighbors (KNN)
5. SVM - Support Vector Machine

### 📈 Como Avaliamos

Não dá pra confiar só na "acurácia", né? Logo, usamos várias métricas:
- **Accuracy** → quantos acertamos no geral
- **Precision** → quando dizemos que é maligno, quantos realmente são?
- **Recall** → de todos os casos malignos, quantos conseguimos pegar?
- **F1-Score** → um balanço entre precision e recall
- **ROC-AUC** → quão bem o modelo separa as classes

**Recall é a mais importante aqui!** Porque é melhor ter um "alarme falso" do que deixar passar um caso de câncer...

### 🎯 Resultados

Nosso melhor modelo (SVM) conseguiu:
- **97.37%** de acurácia
- **97.67%** de recall (pegou quase todos os casos malignos!)
- **95.45%** de precision

Mais detalhes no notebook e no relatório técnico que fizemos.

### Autores
Grupo Tech Challenge - Fase 1

### Licença
Este projeto é parte do Tech Challenge da Pós-Tech FIAP.
