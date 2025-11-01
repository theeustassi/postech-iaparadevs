# Instruções de Execução - Tech Challenge EXTRA

## Guia Passo a Passo para Executar o Projeto

Este guia detalha todos os passos necessários para executar o projeto de detecção de pneumonia em raios-X.

---

## Pré-requisitos

Antes de começar, certifique-se de ter:

- Python 3.9 ou superior instalado
- Git instalado
- Conta no Kaggle (para baixar o dataset)
- Pelo menos 10 GB de espaço livre em disco
- (Opcional) GPU com suporte CUDA para treinamento mais rápido

---

## Passo 1: Clonar o Repositório

```powershell
# Clone o repositório
git clone https://github.com/theeustassi/postech-iaparadevs.git

# Entre no diretório do projeto extra
cd postech-iaparadevs/Tech_Challenge_Extra
```

---

## Passo 2: Configurar Ambiente Python

### Opção A: Ambiente Virtual (Recomendado)

```powershell
# Criar ambiente virtual
python -m venv venv

# Ativar ambiente virtual
.\venv\Scripts\Activate.ps1

# Você verá (venv) no início do prompt

# Instalar dependências
pip install --upgrade pip
pip install -r requirements.txt
```

### Opção B: Conda

```powershell
# Criar ambiente conda
conda create -n pneumonia python=3.9

# Ativar ambiente
conda activate pneumonia

# Instalar dependências
pip install -r requirements.txt
```

---

## 🔑 Passo 3: Configurar Credenciais do Kaggle

### 3.1 Obter API Token

1. Acesse https://www.kaggle.com
2. Faça login na sua conta
3. Clique no seu perfil (canto superior direito)
4. Vá em **Account**
5. Role até a seção **API**
6. Clique em **Create New API Token**
7. Um arquivo `kaggle.json` será baixado

### 3.2 Configurar o Token

**No Windows:**
```powershell
# Criar diretório .kaggle se não existir
mkdir $env:USERPROFILE\.kaggle -Force

# Copiar o arquivo kaggle.json para lá
copy caminho\onde\baixou\kaggle.json $env:USERPROFILE\.kaggle\

# Verificar
ls $env:USERPROFILE\.kaggle\
```

**No Linux/Mac:**
```bash
mkdir -p ~/.kaggle
cp /caminho/para/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

---

## Passo 4: Baixar o Dataset

```powershell
# Executar script de download
python src/download_dataset.py
```

**O que esperar:**
- Download de ~2 GB de dados
- Descompactação automática
- Verificação da estrutura do dataset
- Tempo estimado: 5-15 minutos (depende da internet)

**Saída esperada:**
```
Sistema de Detecção de Pneumonia - Download do Dataset
============================================================
Baixando dataset do Kaggle...
Isso pode levar alguns minutos dependendo da sua conexão...
Baixando paultimothymooney/chest-xray-pneumonia...
Download concluído!

Estrutura do dataset:
  TRAIN:
    - NORMAL: 1341 imagens
    - PNEUMONIA: 3875 imagens
  TEST:
    - NORMAL: 234 imagens
    - PNEUMONIA: 390 imagens
  VAL:
    - NORMAL: 8 imagens
    - PNEUMONIA: 8 imagens

Dataset pronto para uso em: data/chest_xray
============================================================
Processo finalizado!
```

---

## Passo 5: Executar Análise Exploratória

```powershell
# Iniciar Jupyter Notebook
jupyter notebook notebooks/
```

**No navegador que abrir:**

1. Abra o notebook `01_exploracao_dados.ipynb`
2. Execute célula por célula (Shift + Enter)
3. Ou execute todas: **Cell > Run All**

**O que você verá:**
- Distribuição das classes
- Gráficos de barras e pizza
- Amostras de imagens
- Análise de dimensões
- Distribuição de intensidade dos pixels

**Tempo estimado:** 5-10 minutos

---

## Passo 6: Treinar os Modelos

### 6.1 Abrir Notebook de Treinamento

No Jupyter Notebook, abra: `02_treinamento_modelo.ipynb`

### 6.2 Executar Treinamento

**Opção 1: Executar tudo de uma vez (recomendado para quem tem tempo)**

```
Cell > Run All
```

Tempo estimado:
- Com GPU: 30-60 minutos
- Com CPU: 2-4 horas

**Opção 2: Executar modelo por modelo (recomendado para iniciantes)**

Execute as células sequencialmente:

1. **Seção 1-4**: Setup e preparação (5 min)
2. **Seção Modelo 1**: CNN Simples (10-30 min)
3. **Seção Modelo 2**: VGG16 (15-45 min)
4. **Seção Modelo 3**: ResNet50 (15-45 min)
5. **Seção Final**: Comparação e análise (2 min)

### 6.3 O que acontece durante o treinamento

```
Época 1/20
163/163 [==============================] - 120s 735ms/step - loss: 0.4521 - accuracy: 0.8234
Época 2/20
163/163 [==============================] - 118s 725ms/step - loss: 0.2891 - accuracy: 0.8876
...
```

**Indicadores:**
- `loss`: Quanto menor, melhor (o modelo está aprendendo)
- `accuracy`: Quanto maior, melhor (% de acertos)
- `val_loss` / `val_accuracy`: Métricas na validação

---

## Passo 7: Analisar Resultados

Após o treinamento, você terá:

### 7.1 Arquivos Gerados

```
results/
├── graficos/
│   ├── cnn_simple_history.png           # Histórico CNN Simples
│   ├── cnn_simple_confusion_matrix.png  # Matriz de confusão
│   ├── cnn_simple_roc_curve.png         # Curva ROC
│   ├── vgg16_history.png                # Histórico VGG16
│   ├── vgg16_confusion_matrix.png
│   ├── vgg16_roc_curve.png
│   ├── resnet50_history.png             # Histórico ResNet50
│   ├── resnet50_confusion_matrix.png
│   └── resnet50_roc_curve.png
├── modelos/
│   ├── cnn_simple_best.h5               # Modelo CNN salvo
│   ├── vgg16_transfer_best.h5           # Modelo VGG16 salvo
│   └── resnet50_transfer_best.h5        # Modelo ResNet50 salvo
├── comparacao_modelos.csv               # Tabela comparativa
└── resumo_resultados.txt                # Resumo textual
```

### 7.2 Visualizar Resultados

**No notebook:**
- Todos os gráficos são exibidos inline
- Tabela de comparação final
- Análise detalhada

**Nos arquivos:**
- Abra as imagens PNG para visualização
- Leia o `resumo_resultados.txt` para overview
- Abra `comparacao_modelos.csv` no Excel

---

## 🔍 Passo 8: Interpretar os Resultados

### 8.1 Entender as Métricas

| Métrica | O que significa | Ideal |
|---------|-----------------|-------|
| **Accuracy** | % de acertos geral | > 90% |
| **Precision** | Dos que disse PNEUMONIA, quantos realmente são? | > 85% |
| **Recall** | Dos que TÊM pneumonia, quantos detectei? | > 95% ⚠️ CRÍTICO |
| **F1-Score** | Equilíbrio entre Precision e Recall | > 0.90 |
| **AUC-ROC** | Capacidade de discriminação | > 0.95 |

### 8.2 Analisar Matriz de Confusão

```
                Predito
            NORMAL | PNEUMONIA
Real -------------------------
NORMAL   |   TN   |    FP    |  ← Falsos positivos (ok)
PNEUMONIA|   FN   |    TP    |  ← Falsos negativos (CRÍTICO!)
         -------------------------
```

**Foco principal:** Minimizar FN (Falsos Negativos)

### 8.3 Melhor Modelo

O notebook mostrará qual modelo teve melhor desempenho em cada métrica.

**Exemplo de saída:**
```
🏆 MELHORES MODELOS POR MÉTRICA:
====================================
🎯 Melhor Accuracy:  ResNet50 (93.75%)
🎯 Melhor F1-Score:  ResNet50 (0.9421)
🎯 Melhor AUC-ROC:   ResNet50 (0.9789)
```

---

## 🐳 Passo 9: Executar com Docker (Opcional)

Se preferir usar Docker:

```powershell
# Build da imagem
docker build -t pneumonia-detector .

# Executar container
docker run -p 8888:8888 -v ${PWD}:/workspace pneumonia-detector
```

Acesse: http://localhost:8888
