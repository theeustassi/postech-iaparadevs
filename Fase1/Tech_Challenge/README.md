# Sistema de Apoio ao Diagnóstico de Câncer de Mama

## Tech Challenge - Fase 1 | FIAP Pós-Tech IA para Devs

Olá! Bem-vindo ao meu projeto!

Desenvolvi este sistema como parte do Tech Challenge da FIAP. A ideia aqui é usar Machine Learning para ajudar no diagnóstico de câncer de mama - classificando tumores como benignos ou malignos com base em características extraídas de exames de biópsia.

Basicamente treinei alguns modelos para analisar dados de células e tentar prever se um tumor é perigoso ou não. É tipo ter um assistente inteligente que olha os dados e dá uma opinião baseada em padrões que aprendeu de casos anteriores.

## O que foi feito aqui?

Este sistema:
- Analisa 30 características diferentes das células (tamanho, textura, formato, etc.)
- Testa 5 algoritmos diferentes de ML para ver qual funciona melhor
- Explica as decisões usando SHAP (para entender o "porquê" por trás das previsões)
- Gera gráficos e visualizações dos resultados

> **Obs importante**: Isso é um trabalho acadêmico! Na vida real, a decisão final sempre tem que ser de um médico de verdade. Este modelo é apenas uma ferramenta de apoio mesmo.

### Estrutura do Projeto
```
Tech_Challenge/
├── notebooks/
│   └── diagnostico_cancer_mama.ipynb
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── models.py
│   ├── evaluation.py
│   ├── main.py
│   └── __pycache__/
├── results/
│   ├── graficos/
│   └── modelos treinados (.pkl)
├── Dockerfile
├── requirements.txt
├── RELATORIO_TECNICO.md
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

### Sobre os Dados

Usei o **Wisconsin Breast Cancer Dataset** - um dataset bem conhecido na área:
- **Fonte**: Vem direto da biblioteca **scikit-learn** (integrado no pacote)
- **Origem Original**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+%28diagnostic%29) / [Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
- **Como acessar**: Usa a função `load_breast_cancer()` do sklearn - não requer download de arquivo
- **Conteúdo**: 569 casos com 30 características numéricas de cada tumor
- **Características**: Raio médio, textura, perímetro, área, suavidade, compacidade... tudo extraído de imagens de biópsias
- **Objetivo**: Classificar se é maligno (M) ou benigno (B)

#### Principais Correlações com o Diagnóstico
As features mais relacionadas com tumores malignos são:
1. **worst concave points** (0.7936) - Pontos côncavos no "pior" cenário
2. **worst perimeter** (0.7829) - Perímetro maior
3. **mean concave points** (0.7766) - Pontos côncavos na média
4. **worst radius** (0.7765) - Raio maior
5. **mean perimeter** (0.7426) - Perímetro médio maior

Tumores malignos tendem a ser **maiores** (radius, perimeter, area) e com **bordas mais irregulares** (concave points).

### Ferramentas que Usei

Trabalhei com Python e várias bibliotecas:
- **Pandas** e **NumPy** → para manipular os dados
- **Scikit-learn** → para os algoritmos de ML
- **Matplotlib/Seaborn** → gerar os gráficos
- **SHAP** → entender o que o modelo está "pensando"
- **Jupyter Notebook** → onde fiz tudo de forma interativa

### Modelos que Testei

Rodei 5 algoritmos diferentes pra ver qual seria melhor:
1. Regressão Logística (o mais simples)
2. Árvore de Decisão
3. Random Forest
4. K-Nearest Neighbors (KNN)
5. SVM - Support Vector Machine

### Como Avaliei

Não dá pra confiar só na "acurácia". Logo, usei várias métricas:
- **Accuracy** → quantos acertamos no geral
- **Precision** → quando dizemos que é maligno, quantos realmente são?
- **Recall** → de todos os casos malignos, quantos consegui pegar?
- **F1-Score** → um balanço entre precision e recall
- **ROC-AUC** → quão bem o modelo separa as classes

**Recall é a mais importante aqui!** Porque é melhor ter um "alarme falso" do que deixar passar um caso de câncer...

### Resultados

Meu melhor modelo conseguiu:
- **Accuracy**: 97.37% - Taxa geral de acerto
- **Precision**: 97.56% - Quando diz que é maligno, está certo 97.56% das vezes
- **Recall**: 95.24% - Consegui pegar 95.24% dos casos malignos! (apenas 2 em 113 foram perdidos)
- **F1-Score**: 96.39% - Excelente balanço entre precision e recall
- **ROC-AUC**: 99.54% - Separação quase perfeita entre as classes

**Matriz de Confusão no Conjunto de Teste**:
```
              Predito
           Benigno  Maligno
Real ┌─────────────────────┐
Ben. │   71    │    1      │  (71 corretos, 1 falso alarme)
Mal. │    2    │   40      │  (40 corretos, 2 perdidos)
     └─────────────────────┘
```

Mais detalhes (incluindo análise SHAP e comparação Mean vs Worst features) no notebook e no relatório técnico que desenvolvi.

### Autor
Matheus Tassi Souza - RM367424