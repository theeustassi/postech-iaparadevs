# 📄 Relatório Técnico - Sistema de Detecção de Pneumonia

**Tech Challenge EXTRA - Fase 1**  
**Disciplina:** IA para Devs - FIAP Pós-Tech  
**Projeto:** Sistema de Detecção de Pneumonia em Raios-X usando Visão Computacional

---

## 📋 1. Resumo Executivo

Este projeto implementa um sistema de classificação de imagens médicas utilizando **Redes Neurais Convolucionais (CNN)** para detectar pneumonia em radiografias de tórax. O sistema foi desenvolvido como atividade extra do Tech Challenge, aplicando conceitos de **Visão Computacional** e **Deep Learning** para resolver um problema real da área médica.

### Principais Resultados:
- ✅ Implementação de 3 arquiteturas diferentes de CNN
- ✅ Utilização de Transfer Learning para otimização
- ✅ **Accuracy de 87.82%** no conjunto de teste (ResNet50)
- ✅ **AUC-ROC de 0.9538** (CNN Simples)
- ✅ Sistema de apoio à decisão médica funcional

> ⚠️ **Nota Importante**: Este é um sistema de apoio à decisão. O diagnóstico final deve sempre ser realizado por profissionais médicos qualificados.

---

## 🎯 2. Objetivo do Projeto

### Objetivo Geral
Desenvolver um sistema inteligente capaz de classificar radiografias de tórax em duas categorias:
- **NORMAL**: Pulmões saudáveis
- **PNEUMONIA**: Presença de pneumonia

### Objetivos Específicos
1. Implementar pipeline completo de pré-processamento de imagens médicas
2. Treinar múltiplas arquiteturas de CNN (baseline e transfer learning)
3. Avaliar modelos com métricas apropriadas para contexto médico
4. Comparar desempenho e selecionar melhor modelo
5. Documentar todo o processo de forma didática

---

## 📊 3. Dataset

### 3.1 Fonte dos Dados

**Nome**: Chest X-Ray Images (Pneumonia)  
**Fonte**: [Kaggle - Paul Mooney](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)  
**Descrição**: Dataset contendo 5,863 radiografias de tórax em formato JPEG, organizadas em categorias NORMAL e PNEUMONIA.

### 3.2 Estrutura do Dataset

```
chest_xray/
├── train/              # Conjunto de treinamento
│   ├── NORMAL/         # ~1,341 imagens
│   └── PNEUMONIA/      # ~3,875 imagens
├── test/               # Conjunto de teste
│   ├── NORMAL/         # ~234 imagens
│   └── PNEUMONIA/      # ~390 imagens
└── val/                # Conjunto de validação
    ├── NORMAL/         # ~8 imagens
    └── PNEUMONIA/      # ~8 imagens
```

### 3.3 Características dos Dados

**Distribuição:**
- Total de imagens: 5,863
- Training: ~5,216 imagens (89%)
- Validation: ~16 imagens (0.3%)
- Test: ~624 imagens (10.7%)

**Observações Importantes:**
1. **Desbalanceamento**: Aproximadamente 3:1 (PNEUMONIA:NORMAL)
2. **Dimensões variadas**: Imagens têm diferentes tamanhos (necessário redimensionamento)
3. **Escala de cinza**: Radiografias em tons de cinza (1 canal)
4. **Qualidade**: Imagens de boa qualidade, mas com variação de contraste

---

## 🔧 4. Metodologia

### 4.1 Pré-processamento de Dados

#### 4.1.1 Redimensionamento
- **Tamanho padrão**: 224x224 pixels
- **Justificativa**: Tamanho aceito por modelos pré-treinados (ImageNet)
- **Método**: Interpolação bilinear (cv2.resize)

#### 4.1.2 Normalização
- **Técnica**: Divisão por 255
- **Resultado**: Pixels com valores entre 0 e 1
- **Justificativa**: Facilita convergência do treinamento

#### 4.1.3 Data Augmentation
Para aumentar a variedade e robustez do modelo, aplicamos:

| Técnica | Parâmetro | Justificativa |
|---------|-----------|---------------|
| Rotação | ±15° | Raios-X podem ter leve rotação |
| Deslocamento horizontal | 10% | Posicionamento do paciente varia |
| Deslocamento vertical | 10% | Variação na captura da imagem |
| Cisalhamento | 10% | Pequenas distorções são comuns |
| Zoom | ±10% | Diferentes distâncias da fonte |
| Espelhamento horizontal | Sim | Pulmões são simétricos |

**Implementação:**
```python
ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)
```

#### 4.1.4 Tratamento de Desbalanceamento
Utilizamos **class weights** para penalizar mais os erros na classe minoritária:
```python
class_weight = {
    0: peso_calculado_normal,
    1: peso_calculado_pneumonia
}
```

### 4.2 Arquiteturas de Modelos

Implementamos três arquiteturas distintas para comparação:

#### 4.2.1 Modelo 1: CNN Simples (Baseline)

**Arquitetura:**
```
Input (224x224x3)
    ↓
Conv2D(32, 3x3) + ReLU → MaxPooling → BatchNorm
    ↓
Conv2D(64, 3x3) + ReLU → MaxPooling → BatchNorm
    ↓
Conv2D(128, 3x3) + ReLU → MaxPooling → BatchNorm
    ↓
Conv2D(128, 3x3) + ReLU → MaxPooling → BatchNorm
    ↓
Flatten
    ↓
Dense(512) + ReLU → Dropout(0.5)
    ↓
Dense(256) + ReLU → Dropout(0.5)
    ↓
Dense(1, sigmoid)
```

**Parâmetros:**
- Total de parâmetros: ~7.5M
- Parâmetros treináveis: ~7.5M
- Learning rate inicial: 0.001

**Justificativa:**
- Modelo baseline para estabelecer linha de base
- Arquitetura clássica de CNN
- Boa para entender conceitos fundamentais

#### 4.2.2 Modelo 2: VGG16 Transfer Learning

**Arquitetura:**
```
VGG16 Base (pré-treinada, camadas congeladas)
    ↓
GlobalAveragePooling2D
    ↓
Dense(256) + ReLU → Dropout(0.5)
    ↓
Dense(128) + ReLU → Dropout(0.3)
    ↓
Dense(1, sigmoid)
```

**Configuração:**
- Base: VGG16 pré-treinada no ImageNet
- Camadas treináveis: últimas 4 camadas
- Learning rate: 0.0001 (menor devido ao fine-tuning)

**Justificativa:**
- Aproveita conhecimento de ImageNet
- VGG16 é eficaz em tarefas de visão computacional
- Fine-tuning seletivo reduz tempo de treinamento

#### 4.2.3 Modelo 3: ResNet50 Transfer Learning

**Arquitetura:**
```
ResNet50 Base (pré-treinada)
    ↓
GlobalAveragePooling2D
    ↓
Dense(512) + ReLU → BatchNorm → Dropout(0.5)
    ↓
Dense(256) + ReLU → BatchNorm → Dropout(0.3)
    ↓
Dense(1, sigmoid)
```

**Configuração:**
- Base: ResNet50 pré-treinada no ImageNet
- Camadas treináveis: últimas 10 camadas
- Learning rate: 0.0001

**Justificativa:**
- Arquitetura mais moderna com skip connections
- Melhor desempenho em tarefas complexas
- Residual learning facilita treinamento profundo

### 4.3 Treinamento

#### Hiperparâmetros Globais
```python
BATCH_SIZE = 32
EPOCHS = 20 (máximo, com early stopping)
OPTIMIZER = Adam
LOSS = binary_crossentropy
```

#### Callbacks Utilizados

**1. Early Stopping**
```python
EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)
```
- Para treinamento se não houver melhora por 5 épocas
- Restaura pesos da melhor época

**2. Model Checkpoint**
```python
ModelCheckpoint(
    filepath='modelo_best.h5',
    monitor='val_loss',
    save_best_only=True
)
```
- Salva apenas o melhor modelo

**3. Reduce Learning Rate on Plateau**
```python
ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-7
)
```
- Reduz LR quando estagnado
- Ajuda a convergir melhor

### 4.4 Métricas de Avaliação

Utilizamos múltiplas métricas devido à criticidade do contexto médico:

#### 4.4.1 Métricas Principais

**1. Accuracy (Acurácia)**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```
- Percentual geral de acertos
- Pode ser enganosa com dados desbalanceados

**2. Precision (Precisão)**
```
Precision = TP / (TP + FP)
```
- Dos casos diagnosticados como PNEUMONIA, quantos realmente são?
- Minimiza falsos positivos

**3. Recall (Sensibilidade)**
```
Recall = TP / (TP + FN)
```
- Dos casos reais de PNEUMONIA, quantos detectamos?
- **CRÍTICO EM MEDICINA** - minimiza falsos negativos

**4. F1-Score**
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```
- Média harmônica entre Precision e Recall
- Equilibra ambas as métricas

**5. AUC-ROC**
- Área sob a curva ROC
- Mede capacidade de discriminação
- Independente do threshold

#### 4.4.2 Matriz de Confusão

```
                    Predito
                NORMAL | PNEUMONIA
            ----------------------
Real NORMAL    |  TN   |   FP    |
     PNEUMONIA |  FN   |   TP    |
            ----------------------
```

- **TN (True Negative)**: Correto - NORMAL como NORMAL
- **FP (False Positive)**: Erro - NORMAL como PNEUMONIA (Alarme falso)
- **FN (False Negative)**: Erro - PNEUMONIA como NORMAL ⚠️ **CRÍTICO**
- **TP (True Positive)**: Correto - PNEUMONIA como PNEUMONIA

### 4.5 Ambiente de Desenvolvimento

**Hardware:**
- CPU: [Especificar]
- RAM: [Especificar]
- GPU: [Especificar se disponível]

**Software:**
- Python: 3.9+
- TensorFlow: 2.13.0
- Keras: 2.13.1
- NumPy, Pandas, Matplotlib, Seaborn
- Jupyter Notebook

**Tempo de Treinamento Estimado:**
- CNN Simples: ~15-30 minutos (CPU) / ~5-10 min (GPU)
- VGG16: ~30-60 minutos (CPU) / ~10-20 min (GPU)
- ResNet50: ~30-60 minutos (CPU) / ~10-20 min (GPU)

---

## 📈 5. Resultados

### 5.1 Comparação dos Modelos

**Resultados Obtidos no Treinamento:**

| Modelo | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|--------|----------|-----------|--------|----------|---------|
| **CNN Simples** | 79.17% | 75.29% | 99.23% | 0.8562 | **0.9538** ⭐ |
| **VGG16** | 62.50% | 62.50% | 100.00% | 0.7692 | 0.6141 |
| **ResNet50** | **87.82%** ⭐ | **88.67%** ⭐ | 92.31% | **0.9045** ⭐ | 0.9296 |

**Legenda:** ⭐ = Melhor resultado na métrica

### 5.2 Análise dos Resultados

#### 5.2.1 CNN Simples (Baseline)
**Desempenho Obtido:**
- Accuracy: 79.17%
- **Recall: 99.23%** (Excelente! Detecta quase todos os casos de pneumonia)
- **AUC-ROC: 0.9538** (Melhor capacidade de discriminação entre classes)
- Precision: 75.29% (mais falsos positivos)

**Pontos Fortes:**
- ✅ Rápido para treinar (~10 minutos com GPU)
- ✅ Fácil de entender e depurar
- ✅ **Melhor Recall** (99.23% - crítico em medicina!)
- ✅ **Melhor AUC-ROC** (0.9538)
- ✅ Baseline sólido

**Pontos Fracos:**
- ⚠️ Mais falsos positivos (Precision 75.29%)
- ⚠️ Accuracy inferior ao ResNet50

#### 5.2.2 VGG16 Transfer Learning
**Desempenho Obtido:**
- Accuracy: 62.50% (Pior desempenho)
- Recall: 100.00%
- Precision: 62.50%
- AUC-ROC: 0.6141 (muito baixo)

**⚠️ Problema Crítico Identificado:**
- O modelo está **classificando TUDO como PNEUMONIA**
- Precision para classe NORMAL: 0.00% (não detecta nenhum caso normal corretamente)
- Recall 100% não é positivo neste caso (classifica tudo como uma classe)

**Possíveis Causas:**
- ❌ Desbalanceamento não tratado adequadamente
- ❌ Learning rate pode estar muito alto
- ❌ Fine-tuning insuficiente das camadas
- ❌ Modelo pode ter convergido para mínimo local

**Pontos Fracos:**
- ❌ Pior desempenho entre todos os modelos
- ❌ Não consegue distinguir entre classes
- ❌ Muitos parâmetros (14.8M) mas performance ruim
- ❌ Necessita de retreinamento com ajustes

#### 5.2.3 ResNet50 Transfer Learning
**Desempenho Obtido:**
- **Accuracy: 87.82%** ⭐ (Melhor!)
- **Precision: 88.67%** ⭐ (Melhor!)
- Recall: 92.31% (Excelente!)
- **F1-Score: 0.9045** ⭐ (Melhor!)
- AUC-ROC: 0.9296 (Muito bom)

**Análise Detalhada:**
- Para NORMAL: Precision=86.24%, Recall=80.34%
- Para PNEUMONIA: Precision=88.67%, Recall=92.31%
- **Melhor equilíbrio entre todas as métricas**

**Pontos Fortes:**
- ✅ **Melhor Accuracy, Precision e F1-Score**
- ✅ Arquitetura moderna com skip connections
- ✅ Excelente capacidade de generalização
- ✅ Recall alto (92.31%) - importante para medicina
- ✅ Bom equilíbrio entre Precision e Recall
- ✅ Menos propenso a vanishing gradient

**Pontos Fracos:**
- ⚠️ Mais complexo de entender
- ⚠️ Requer mais recursos computacionais
- ⚠️ AUC-ROC ligeiramente inferior à CNN Simples

### 5.3 Melhor Modelo

**🏆 Modelo Recomendado: ResNet50 Transfer Learning**

**Justificativa:**

1. **Melhor Performance Geral:**
   - Accuracy: 87.82% (8.65 pontos percentuais acima da CNN Simples)
   - F1-Score: 0.9045 (melhor equilíbrio Precision/Recall)
   - Precision: 88.67% (reduz falsos positivos)

2. **Contexto Médico:**
   - Recall de 92.31% é excelente (detecta 92% dos casos de pneumonia)
   - Precision alta (88.67%) reduz falsos alarmes
   - Melhor equilíbrio para uso clínico real

3. **Confiabilidade:**
   - AUC-ROC de 0.9296 indica excelente capacidade discriminativa
   - Modelo mais robusto e generaliza melhor
   - Arquitetura moderna e bem validada

**📊 Comparação com CNN Simples:**

| Aspecto | CNN Simples | ResNet50 | Vantagem |
|---------|-------------|----------|----------|
| Recall | 99.23% ⭐ | 92.31% | CNN (~7% mais casos detectados) |
| Precision | 75.29% | 88.67% ⭐ | ResNet (~13% menos falsos positivos) |
| Accuracy | 79.17% | 87.82% ⭐ | ResNet (+8.65%) |
| F1-Score | 0.8562 | 0.9045 ⭐ | ResNet (melhor equilíbrio) |

**💡 Quando usar cada modelo:**

- **ResNet50**: Uso geral em ambiente clínico (recomendado)
- **CNN Simples**: Quando recall absoluto é crítico (triagem inicial, não perder nenhum caso)
- **VGG16**: Necessita retreinamento (não recomendado no estado atual)

---

## 💭 6. Discussão

### 6.1 Interpretação dos Resultados

#### Contexto Médico
Em aplicações médicas, a **sensibilidade (Recall)** é tipicamente mais importante que a precisão:

- **Falso Positivo (FP)**: Paciente saudável diagnosticado com pneumonia
  - Consequência: Exames adicionais desnecessários
  - Impacto: Médio (custo e ansiedade)
  
- **Falso Negativo (FN)**: Paciente com pneumonia não detectado
  - Consequência: Doença não tratada
  - Impacto: **ALTO** (risco à vida)

Portanto, é preferível ter alguns falsos positivos do que deixar passar casos reais de pneumonia.

### 6.2 Insights dos Resultados Obtidos

**🔍 Análise Comparativa:**

1. **Transfer Learning funcionou melhor:**
   - ResNet50 (87.82%) >> CNN Simples (79.17%)
   - Conhecimento pré-treinado do ImageNet ajudou significativamente
   - Skip connections do ResNet previnem vanishing gradient

2. **Trade-off Precision vs Recall:**
   - CNN Simples: Recall alto (99.23%), mas Precision baixa (75.29%)
   - ResNet50: Melhor equilíbrio (Recall 92.31%, Precision 88.67%)
   - Para triagem: CNN Simples é melhor (não perde casos)
   - Para diagnóstico: ResNet50 é melhor (menos falsos alarmes)

3. **Problema do VGG16:**
   - Classificou TUDO como PNEUMONIA (Precision NORMAL = 0%)
   - Possível causa: desbalanceamento não tratado adequadamente
   - Learning rate ou número de camadas treináveis pode estar inadequado
   - Demonstra importância de validação cuidadosa

4. **AUC-ROC Insights:**
   - CNN Simples tem melhor AUC (0.9538) que ResNet50 (0.9296)
   - Indica que CNN Simples tem melhor capacidade de separação entre classes
   - Mas accuracy menor sugere threshold de decisão subótimo

**💡 Lições Aprendidas:**

- ✅ Class weights são essenciais em datasets desbalanceados
- ✅ Early stopping preveniu overfitting (parou em ~12 épocas)
- ✅ Data augmentation ajudou na generalização
- ⚠️ Nem sempre mais parâmetros = melhor resultado (VGG16 falhou)
- ⚠️ Fine-tuning requer ajuste cuidadoso de hiperparâmetros

### 6.3 Limitações do Estudo

1. **Tamanho do Dataset**: 
   - Dataset relativamente pequeno para deep learning (~5,200 imagens de treino)
   - Possível overfitting mesmo com data augmentation
   - Comparação: modelos modernos usam milhões de imagens

2. **Desbalanceamento**:
   - Proporção 3:1 de PNEUMONIA:NORMAL
   - Class weights ajudaram, mas VGG16 ainda teve problemas
   - Dataset de validação muito pequeno (16 imagens)

3. **Generalização**:
   - Dataset de uma única fonte (Kaggle)
   - Pode não generalizar para diferentes equipamentos de raio-X
   - Diferentes hospitais têm protocolos de imagem diferentes

4. **Tipos de Pneumonia**:
   - Dataset não distingue entre pneumonia viral e bacteriana
   - Não identifica outros problemas pulmonares (tuberculose, câncer)
   - Informação clínica relevante não capturada

5. **Validação Clínica**:
   - Necessária validação com médicos radiologistas
   - Testes em ambiente clínico real
   - Análise de casos limítrofes e difíceis

### 6.4 Possibilidade de Uso Prático

**✅ Pode ser usado como:**
- Ferramenta de **triagem** inicial
- Sistema de **apoio à decisão** médica
- Ferramenta de **segunda opinião**
- Sistema de **alerta** para casos suspeitos

**❌ NÃO deve ser usado como:**
- Substituto do diagnóstico médico
- Única fonte de decisão clínica
- Ferramenta sem supervisão profissional

**Recomendação de Uso:**
```
1. Sistema faz análise automática
2. Casos suspeitos são sinalizados
3. Médico revisa TODOS os casos
4. Decisão final é sempre do médico
5. Sistema aprende com feedback
```

---

## 🚀 7. Conclusões

### 7.1 Objetivos Alcançados

✅ **Implementação técnica completa:**
- Pipeline de pré-processamento robusto
- Três arquiteturas diferentes de CNN
- Sistema de avaliação abrangente
- Documentação detalhada

✅ **Resultados satisfatórios:**
- **ResNet50**: 87.82% de accuracy, modelo recomendado
- **CNN Simples**: 99.23% de recall, excelente para triagem
- **VGG16**: 62.50% de accuracy, necessita ajustes
- Métricas adequadas para contexto médico
- Comparação sistemática entre abordagens

✅ **Aprendizado prático:**
- Visão computacional aplicada
- Transfer learning
- Deep learning para imagens médicas
- Avaliação crítica de modelos

### 7.2 Contribuições do Projeto

1. **Técnicas:**
   - Demonstração de Transfer Learning em medicina
   - Pipeline reproduzível para classificação de imagens médicas
   - Boas práticas de data augmentation

2. **Educacionais:**
   - Código bem documentado para iniciantes
   - Explicações didáticas de conceitos
   - Notebooks interativos

3. **Práticas:**
   - Sistema funcional de apoio ao diagnóstico
   - Framework extensível para outras doenças
   - Base para projetos futuros

### 7.3 Trabalhos Futuros

**Melhorias Imediatas:**
- [ ] Testar outras arquiteturas (EfficientNet, DenseNet)
- [ ] Implementar Grad-CAM para explicabilidade
- [ ] Ajustar hiperparâmetros com GridSearch
- [ ] Ensemble de modelos

**Expansões:**
- [ ] Classificar tipos de pneumonia (viral vs bacteriana)
- [ ] Detectar outras doenças pulmonares
- [ ] Multi-class classification
- [ ] Segmentação de regiões afetadas

**Deploy:**
- [ ] API REST com FastAPI
- [ ] Interface web interativa
- [ ] Aplicativo mobile
- [ ] Integração com PACS (Picture Archiving and Communication System)

**Validação:**
- [ ] Testes com médicos radiologistas
- [ ] Validação cruzada com outros datasets
- [ ] Estudo de caso em ambiente clínico

---

## 📚 8. Referências

### Dataset
1. Paul Mooney. (2018). *Chest X-Ray Images (Pneumonia)*. Kaggle. https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

### Frameworks e Bibliotecas
7. TensorFlow: https://www.tensorflow.org/
8. Keras: https://keras.io/
9. Scikit-learn: https://scikit-learn.org/

---

## 👥 9. Informações do Projeto

**Instituição:** FIAP - Faculdade de Informática e Administração Paulista  
**Curso:** Pós-Tech - IA para Devs  
**Fase:** 1  

---

## 📝 10. Apêndices

### Apêndice A: Estrutura de Arquivos

```
Tech_Challenge_Extra/
├── README.md                           # Documentação principal
├── RELATORIO_TECNICO.md               # Este arquivo
├── requirements.txt                    # Dependências Python
├── Dockerfile                          # Container Docker
├── data/
│   └── chest_xray/                    # Dataset (não versionado)
├── notebooks/
│   ├── 01_exploracao_dados.ipynb      # Análise exploratória
│   └── 02_treinamento_modelo.ipynb    # Treinamento
├── src/
│   ├── __init__.py
│   ├── download_dataset.py            # Script de download
│   ├── preprocessing.py               # Pré-processamento
│   ├── models.py                      # Arquiteturas CNN
│   └── evaluation.py                  # Avaliação
└── results/
    ├── graficos/                      # Visualizações
    ├── modelos/                       # Modelos salvos
    └── resumo_resultados.txt          # Resumo
```

### Apêndice B: Comandos Úteis

**Setup do Ambiente:**
```powershell
# Criar ambiente virtual
python -m venv venv
.\venv\Scripts\Activate.ps1

# Instalar dependências
pip install -r requirements.txt

# Baixar dataset
python src/download_dataset.py
```

**Executar Notebooks:**
```powershell
jupyter notebook notebooks/
```

**Docker:**
```powershell
docker build -t pneumonia-detector .
docker run -p 8888:8888 pneumonia-detector
```

### Apêndice C: Glossário

- **CNN (Convolutional Neural Network)**: Rede neural especializada em processamento de imagens
- **Transfer Learning**: Técnica de aproveitar modelo pré-treinado
- **Data Augmentation**: Técnica de aumentar dataset com transformações
- **Overfitting**: Modelo se ajusta demais aos dados de treino
- **Batch Size**: Número de amostras processadas por vez
- **Epoch**: Uma passagem completa pelo dataset de treino
- **Learning Rate**: Taxa de aprendizado do modelo
- **Dropout**: Técnica de regularização que desliga neurônios aleatoriamente
- **Fine-tuning**: Ajuste fino de modelo pré-treinado

---

**Fim do Relatório Técnico**

