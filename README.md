# 🎓 Pós-Graduação IA para Devs - FIAP

**Turma:** 7IADT  
**Instituição:** FIAP - Faculdade de Informática e Administração Paulista  
**Programa:** Pós Tech - Inteligência Artificial para Desenvolvedores

---

## 📖 Sobre este Repositório

Este repositório contém todos os exercícios, projetos e atividades desenvolvidos durante a pós-graduação em **Inteligência Artificial para Desenvolvedores** da FIAP. O programa tem como objetivo capacitar desenvolvedores a aplicar técnicas de IA e Machine Learning em projetos reais, com foco prático e hands-on.

## 🎯 Objetivos do Curso

- Dominar fundamentos de Machine Learning e Deep Learning
- Aplicar técnicas de IA em problemas reais de negócio
- Desenvolver modelos preditivos e sistemas inteligentes
- Implementar pipelines completos de ML (da coleta de dados ao deploy)
- Compreender ética e boas práticas em IA

## 📂 Estrutura do Repositório

```
postech-iaparadevs/
│
├── Fase 1/                          # Fundamentos de IA e ML
│   │
│   ├── Tech_Challenge/              # 🏥 Projeto: Diagnóstico de Câncer de Mama
│   │   ├── notebooks/
│   │   │   └── diagnostico_cancer_mama.ipynb
│   │   ├── src/
│   │   │   ├── __init__.py
│   │   │   ├── main.py
│   │   │   ├── preprocessing.py
│   │   │   ├── models.py
│   │   │   └── evaluation.py
│   │   ├── results/
│   │   │   └── graficos/
│   │   ├── .gitignore
│   │   ├── Dockerfile
│   │   ├── README.md
│   │   ├── RELATORIO_TECNICO.md
│   │   ├── INSTRUCOES_ENTREGA.md
│   │   └── requirements.txt
│   │
│   └── Tech_Challenge_Extra/        # 🫁 Projeto Extra: Detecção de Pneumonia
│       ├── notebooks/
│       │   ├── 01_exploracao_dados.ipynb
│       │   └── 02_treinamento_modelo.ipynb
│       ├── src/
│       │   ├── __init__.py
│       │   ├── download_dataset.py
│       │   ├── preprocessing.py
│       │   ├── models.py
│       │   └── evaluation.py
│       ├── data/
│       │   └── chest_xray/          # Dataset Kaggle (5,863 imagens)
│       ├── results/
│       │   ├── graficos/            # Gráficos e visualizações
│       │   ├── modelos/             # Modelos treinados (.h5)
│       │   ├── comparacao_modelos.csv
│       │   └── resumo_resultados.txt
│       ├── .gitignore
│       ├── Dockerfile
│       ├── README.md
│       ├── RELATORIO_TECNICO.md
│       ├── INSTRUCOES_EXECUCAO.md
│       └── requirements.txt
│
└── README.md                        # 📖 Este arquivo (documentação principal)
```

### 📊 Resumo Rápido

| Projeto | Tipo | Tecnologia Principal | Notebooks | Status |
|---------|------|---------------------|-----------|---------|
| **Tech Challenge** | Machine Learning | Scikit-learn | 1 | ✅ Concluído |
| **Tech Challenge Extra** | Deep Learning | TensorFlow/Keras | 2 | ✅ Concluído |

**Observação:** A pasta `data/` não está versionada (incluída no `.gitignore`) devido ao tamanho dos datasets.

---

## 🚀 Projetos em Destaque

### 🏥 Tech Challenge - Diagnóstico de Câncer de Mama
**Tecnologias:** Python, Scikit-learn, Pandas, Matplotlib  
**Descrição:** Sistema de Machine Learning para auxiliar no diagnóstico de câncer de mama usando o dataset Wisconsin Breast Cancer.

**Resultados:**
- Análise exploratória completa
- Modelos de classificação (Logistic Regression, Random Forest, SVM, etc.)
- Métricas de avaliação apropriadas para contexto médico
- Documentação técnica detalhada

📂 [Ver projeto completo](./Fase%201/Tech_Challenge/)

---

### 🫁 Tech Challenge EXTRA - Detecção de Pneumonia por Visão Computacional
**Tecnologias:** Python, TensorFlow, Keras, CNN, Transfer Learning  
**Descrição:** Sistema de Deep Learning para detectar pneumonia em radiografias de tórax usando Redes Neurais Convolucionais.

**Arquiteturas Implementadas:**
- CNN Simples (baseline)
- VGG16 (Transfer Learning)
- ResNet50 (Transfer Learning)

**Resultados Obtidos:**
| Modelo | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|--------|----------|-----------|--------|----------|---------|
| CNN Simples | 79.17% | 75.29% | 99.23% | 0.8562 | 0.9538 |
| ResNet50 | **87.82%** | **88.67%** | 92.31% | **0.9045** | 0.9296 |

**Destaques:**
- ✅ Melhor modelo: ResNet50 com 87.82% de accuracy
- ✅ CNN Simples com recall de 99.23% (excelente para triagem)
- ✅ Pipeline completo de pré-processamento
- ✅ Data augmentation para robustez
- ✅ Avaliação com métricas médicas apropriadas

📂 [Ver projeto completo](./Fase%201/Tech_Challenge_Extra/)

---

## 🛠️ Tecnologias e Ferramentas

### Linguagens
- Python 3.9+

### Machine Learning & Data Science
- **Scikit-learn** - Modelos de ML clássicos
- **TensorFlow / Keras** - Deep Learning
- **Pandas** - Manipulação de dados
- **NumPy** - Computação numérica
- **Matplotlib / Seaborn** - Visualização

### Deep Learning & Computer Vision
- **CNNs** (Redes Neurais Convolucionais)
- **Transfer Learning** (VGG16, ResNet50)
- **Data Augmentation**
- **GPU Computing** (CUDA, cuDNN)

### Desenvolvimento
- **Jupyter Notebooks** - Análise interativa
- **Git / GitHub** - Controle de versão
- **Conda / Pip** - Gerenciamento de pacotes
- **Docker** - Containerização (em alguns projetos)

### Ambiente
- **Python 3.9** (compatibilidade TensorFlow GPU)
- **VS Code** com extensões Python/Jupyter
- **NVIDIA GPU** (RTX 5070 Ti) para treinamento acelerado

---

## 🎓 Habilidades Desenvolvidas

### Machine Learning
- [x] Análise Exploratória de Dados (EDA)
- [x] Pré-processamento de dados (normalização, encoding, etc.)
- [x] Feature Engineering e seleção de features
- [x] Modelos de classificação e regressão
- [x] Validação cruzada e métricas de avaliação
- [x] Tratamento de desbalanceamento de classes

### Deep Learning
- [x] Redes Neurais Convolucionais (CNN)
- [x] Transfer Learning (VGG16, ResNet50)
- [x] Data Augmentation para imagens
- [x] Fine-tuning de modelos pré-treinados
- [x] Callbacks (Early Stopping, ModelCheckpoint, ReduceLROnPlateau)
- [x] Treinamento com GPU (CUDA/cuDNN)

### Visão Computacional
- [x] Pré-processamento de imagens médicas
- [x] Classificação de imagens
- [x] Arquiteturas CNN modernas
- [x] Avaliação de modelos de visão computacional

### Engenharia de Software
- [x] Desenvolvimento de pacotes Python
- [x] Documentação técnica (README, relatórios)
- [x] Versionamento com Git
- [x] Organização de projetos de Data Science
- [x] Criação de pipelines reproduzíveis

### Soft Skills
- [x] Documentação clara e detalhada
- [x] Análise crítica de resultados
- [x] Comunicação técnica
- [x] Ética em IA (contexto médico)

---

## 🚀 Como Usar Este Repositório

### Pré-requisitos
```bash
# Python 3.9+
python --version

# Git
git --version
```

### Clone o Repositório
```bash
git clone https://github.com/theeustassi/postech-iaparadevs.git
cd postech-iaparadevs
```

### Navegue pelos Projetos
Cada projeto tem seu próprio README com instruções específicas:

```bash
# Tech Challenge - ML Clássico
cd "Fase 1/Tech_Challenge"
cat README.md

# Tech Challenge Extra - Deep Learning
cd "../Tech_Challenge_Extra"
cat README.md
```

### Configure o Ambiente (exemplo para Tech Challenge Extra)
```bash
# Crie ambiente conda
conda create -n venv python=3.9 -y
conda activate venv

# Instale dependências
cd "Fase 1/Tech_Challenge_Extra"
pip install -r requirements.txt
```

---

## 📝 Convenções do Repositório

### Documentação
- ✅ Cada projeto tem README próprio com instruções completas
- ✅ Relatórios técnicos detalhados em Markdown
- ✅ Código bem comentado e organizado
- ✅ Docstrings em todas as funções importantes
- ✅ Notebooks com células markdown explicativas

### Commits
- Commits em português
- Mensagens descritivas e claras
- Commits atômicos (uma funcionalidade por vez)
- Formato: `tipo: descrição` (ex: `feat: adiciona modelo ResNet50`)

---

## 👤 Autor

**Estudante:** Matheus Tassi 
**Turma:** 7IADT  
**LinkedIn:**  
**GitHub:** [@theeustassi](https://github.com/theeustassi)

---

## 📄 Licença

Este repositório é para fins educacionais como parte da pós-graduação da FIAP.

**Nota sobre Datasets:**
- Os datasets utilizados são de domínio público ou com licenças permissivas
- Créditos aos criadores estão nos README de cada projeto
- Uso exclusivo para fins acadêmicos

---

<div align="center">

### 🌟 Se este repositório foi útil, considere dar uma estrela! ⭐

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10-orange.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-green.svg)
![Status](https://img.shields.io/badge/Status-Em%20Desenvolvimento-yellow.svg)

</div>
