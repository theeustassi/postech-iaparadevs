# 🫁 Sistema de Detecção de Pneumonia em Raios-X

## Tech Challenge EXTRA - Fase 1 | FIAP Pós-Tech IA para Devs

Bem-vindo ao projeto extra do Tech Challenge! 🎉

Este projeto usa **Visão Computacional** e **Redes Neurais Convolucionais (CNN)** para detectar pneumonia em imagens de raios-X do tórax. É um exemplo prático de como a inteligência artificial pode ajudar profissionais da saúde na análise de exames médicos.

## 🎯 O que esse projeto faz?

O sistema analisa imagens de radiografias de tórax e classifica em duas categorias:
- **NORMAL**: Pulmões saudáveis
- **PNEUMONIA**: Presença de pneumonia

Nós usamos redes neurais profundas que "aprendem" a reconhecer padrões nas imagens, similar a como um radiologista aprende durante anos de prática - mas de forma automatizada!

### Como funciona?

1. 📸 **Recebe**: Uma imagem de raio-X do tórax
2. 🧠 **Processa**: A CNN analisa a imagem em várias camadas, detectando características
3. ✅ **Classifica**: Retorna se é NORMAL ou PNEUMONIA com um nível de confiança

> ⚕️ **IMPORTANTE**: Este é um projeto acadêmico para fins educacionais. Na prática médica real, diagnósticos devem sempre ser realizados por profissionais qualificados. Este sistema serve apenas como ferramenta de apoio à decisão!

## 📊 Dataset

Utilizamos o dataset **Chest X-Ray Images (Pneumonia)** do Kaggle:
- 5,863 imagens de raios-X em formato JPEG
- Divididas em 3 pastas: train, test, val
- Cada pasta contém subpastas NORMAL e PNEUMONIA
- Dataset: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

**Distribuição dos dados:**
- Training: ~5,200 imagens
- Validation: ~16 imagens  
- Test: ~624 imagens

## 🏗️ Estrutura do Projeto

```
Tech_Challenge_Extra/
├── data/
│   └── chest_xray/           # Dataset (baixado via script)
│       ├── train/
│       │   ├── NORMAL/
│       │   └── PNEUMONIA/
│       ├── test/
│       └── val/
├── notebooks/
│   ├── 01_exploracao_dados.ipynb      # Análise exploratória
│   └── 02_treinamento_modelo.ipynb    # Treinamento e avaliação
├── src/
│   ├── __init__.py
│   ├── preprocessing.py      # Pré-processamento de imagens
│   ├── models.py             # Arquiteturas CNN
│   ├── evaluation.py         # Métricas e avaliação
│   └── download_dataset.py   # Script para baixar dados
├── results/
│   ├── graficos/            # Visualizações e gráficos
│   └── modelos/             # Modelos treinados salvos
├── Dockerfile
├── requirements.txt
├── RELATORIO_TECNICO.md
└── README.md
```

## 🚀 Como usar

### Pré-requisitos
- Python 3.9 ou superior
- GPU (opcional, mas recomendada para treinamento mais rápido)
- Conta no Kaggle (para baixar o dataset)

### Opção 1: Instalação Local (Windows)

#### 1. Clone o repositório
```powershell
git clone https://github.com/theeustassi/postech-iaparadevs.git
cd Tech_Challenge_Extra
```

#### 2. Crie um ambiente virtual
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

#### 3. Instale as dependências
```powershell
pip install -r requirements.txt
```

#### 4. Configure o Kaggle API
Baixe suas credenciais do Kaggle:
1. Acesse https://www.kaggle.com/settings
2. Clique em "Create New API Token"
3. Salve o arquivo `kaggle.json` em `~/.kaggle/` (Linux/Mac) ou `C:\Users\<seu_usuario>\.kaggle\` (Windows)

#### 5. Baixe o dataset
```powershell
python src/download_dataset.py
```

#### 6. Execute os notebooks
```powershell
jupyter notebook notebooks/
```

### Opção 2: Usando Docker

```powershell
# Build da imagem
docker build -t pneumonia-detector .

# Executar container
docker run -p 8888:8888 -v ${PWD}:/workspace pneumonia-detector
```

## 🧪 Modelos Implementados

### 1. CNN Simples (Baseline)
- Arquitetura customizada com 3-4 camadas convolucionais
- Boa para entender os conceitos básicos
- ~10-20 minutos de treinamento

### 2. Transfer Learning - VGG16
- Modelo pré-treinado no ImageNet
- Fine-tuning nas últimas camadas
- Melhor precisão com menos tempo de treinamento

### 3. Transfer Learning - ResNet50
- Arquitetura mais moderna com conexões residuais
- Excelente para datasets médicos
- Alto desempenho

## 📈 Métricas de Avaliação

Avaliamos os modelos usando:
- **Accuracy**: Precisão geral
- **Precision**: Quantos dos diagnosticados com pneumonia realmente têm
- **Recall (Sensibilidade)**: Quantos casos de pneumonia conseguimos detectar
- **F1-Score**: Média harmônica entre precision e recall
- **Confusion Matrix**: Visualização de acertos e erros
- **ROC Curve & AUC**: Capacidade de discriminação do modelo

> 💡 No contexto médico, o **Recall** é crítico! É melhor ter alguns falsos positivos (dizer que tem pneumonia quando não tem) do que falsos negativos (não detectar uma pneumonia real).

## 📊 Resultados Obtidos

Treinamos 3 modelos diferentes e obtivemos os seguintes resultados:

| Modelo | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|--------|----------|-----------|--------|----------|---------|
| **CNN Simples** | 79.17% | 75.29% | **99.23%** ⭐ | 85.62% | **0.9538** ⭐ |
| **VGG16** | 62.50% | 62.50% | 100.00% | 76.92% | 0.6141 |
| **ResNet50** | **87.82%** ⭐ | **88.67%** ⭐ | 92.31% | **90.45%** ⭐ | 0.9296 |

### 🏆 Modelo Recomendado: **ResNet50**

**Por quê?**
- ✅ **Melhor accuracy geral**: 87.82%
- ✅ **Melhor precision**: 88.67% (menos falsos positivos)
- ✅ **Excelente recall**: 92.31% (detecta 92% dos casos de pneumonia)
- ✅ **Melhor F1-Score**: 90.45% (melhor equilíbrio)

**CNN Simples também é excelente para:**
- 🎯 **Triagem inicial**: Recall de 99.23% (quase não perde nenhum caso!)
- 🎯 **Melhor AUC-ROC**: 0.9538 (excelente capacidade discriminativa)

**VGG16 teve problemas:**
- ⚠️ Classificou quase tudo como PNEUMONIA
- ⚠️ Precision para NORMAL = 0%
- ⚠️ Necessita ajustes e retreinamento

### 💡 Insights Importantes

1. **Transfer Learning funcionou!** ResNet50 (87.82%) >> CNN Simples (79.17%)
2. **Trade-off Precision vs Recall**: CNN tem recall altíssimo mas mais falsos positivos
3. **Nem sempre mais parâmetros = melhor resultado**: VGG16 (14.8M params) teve pior desempenho
4. **Class weights são essenciais** em datasets desbalanceados
5. **Early stopping preveniu overfitting**: modelos pararam automaticamente em ~12 épocas

Veja análise completa no [RELATORIO_TECNICO.md](RELATORIO_TECNICO.md)!

## 🎓 Conceitos Aprendidos

Este projeto aborda:
- ✅ Processamento de imagens médicas
- ✅ Redes Neurais Convolucionais (CNN)
- ✅ Transfer Learning
- ✅ Data Augmentation
- ✅ Overfitting e técnicas de regularização
- ✅ Métricas para problemas de classificação desbalanceados
- ✅ Visualização de resultados com Grad-CAM

## 📚 Próximos Passos

Ideias para melhorar o projeto:
- [ ] Implementar ensemble de modelos
- [ ] Adicionar explicabilidade com Grad-CAM
- [ ] Testar outras arquiteturas (EfficientNet, DenseNet)
- [ ] Criar API REST para fazer predições
- [ ] Deploy em serviço cloud

## 👥 Contribuidores

[Seu nome aqui]

## 📄 Licença

Este projeto é para fins educacionais - FIAP Pós-Tech IA para Devs

## 🙏 Agradecimentos

- Dataset: Dr. Paul Mooney & Kaggle
- FIAP pela oportunidade de aprendizado
- Comunidade open-source de ML

