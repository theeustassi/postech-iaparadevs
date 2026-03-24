# Guia de Execucao - Tech Challenge Fase 3

## Assistente Medico Virtual HospitalIQ

---

## Pre-requisitos

- Python 3.11+
- Conda (Anaconda ou Miniconda) — recomendado
- 4 GB de RAM (minimo para o modelo flan-t5-base em CPU)
- GPU NVIDIA com CUDA 12.x (opcional, mas recomendado — melhora significativamente a velocidade do fine-tuning)
- Conexao com a internet (para baixar modelos do HuggingFace na primeira execucao)
- API Key do Google Gemini (opcional, mas recomendada para melhor qualidade)

---

## Opcao 1: Script Principal (Recomendado)

Tempo estimado: 5-10 minutos na primeira execucao (baixa os modelos necessarios)

```powershell
# 1. Acesse o diretorio do projeto
cd "postech-iaparadevs\Fase3\Tech_Challenge"

# 2. Crie e ative o ambiente conda
conda create -n fase3 python=3.11 -y
conda activate fase3

# 3. Instale greenlet pre-compilado (necessario no Windows antes do restante)
conda run -n fase3 pip install greenlet --only-binary :all:

# 4. Instale as dependencias
pip install -r requirements.txt

# 5. (Opcional, GPU NVIDIA) Reinstale o PyTorch com suporte CUDA 12.8
pip install torch --force-reinstall --index-url https://download.pytorch.org/whl/cu128

# 6. (Opcional) Configure a API key do Gemini
copy .env.example .env
# Edite o arquivo .env e adicione sua chave GEMINI_API_KEY
# Chave gratuita em: https://makersuite.google.com/app/apikey

# 7. Execute o pipeline completo
python src/main.py
```

### Modos de execucao

```powershell
# Modo demonstracao rapida (sem fine-tuning, apenas o assistente)
python src/main.py --demo-only

# Pipeline completo pulando o fine-tuning (usa modelo base diretamente)
python src/main.py --skip-finetune

# Pipeline completo com fine-tuning (mais lento, recomendado com GPU)
python src/main.py
```

### Arquivos gerados apos a execucao

```
results/
├── data/
│   ├── train.jsonl          # Dataset de treino (450 exemplos, PubMedQA fold 0)
│   ├── val.jsonl            # Dataset de validacao (50 exemplos)
│   └── test.jsonl           # Dataset de teste (500 exemplos)
├── modelos/
│   ├── finetuned_model/     # Modelo fine-tunado com LoRA (se executado)
│   └── vectorstore/         # Base vetorial FAISS dos protocolos
├── logs/
│   └── audit_YYYYMMDD.jsonl # Log de auditoria do dia
├── graficos/                # Visualizacoes geradas (pasta criada automaticamente)
└── evaluation_results.json  # Metricas BLEU, ROUGE, Safety
```

---

## Opcao 2: Jupyter Notebook Interativo

```powershell
# Ative o ambiente conda (se ainda nao estiver ativo)
conda activate fase3

# Abra o Jupyter
jupyter notebook

# Navegue para: notebooks/01_demonstracao_completa.ipynb
# Execute as celulas em ordem
```

---

## Opcao 3: Docker

```powershell
# Construa a imagem
docker build -t hospitaliq-assistant .

# Execute no modo demonstracao
docker run hospitaliq-assistant

# Execute com a API key do Gemini
docker run -e GEMINI_API_KEY=sua_chave_aqui hospitaliq-assistant

# Execute o Jupyter no Docker
docker run -p 8888:8888 hospitaliq-assistant jupyter notebook --ip=0.0.0.0 --no-browser --allow-root
```

---

## Executar os Testes

```powershell
# Todos os testes
pytest tests/ -v

# Testes com relatorio de cobertura
pytest tests/ --cov=src --cov-report=term-missing

# Apenas um modulo especifico
pytest tests/test_assistant.py::TestInputValidator -v
```

Os testes cobrem:
- Anonimizacao de dados (CPF, nomes, datas)
- Validacao e sanitizacao de entradas
- Deteccao de prompt injection
- Classificacao de gravidade na triagem
- Filtragem de prescricoes diretas no safety check
- Logging de auditoria

---

## Resolucao de Problemas Comuns

**Erro: greenlet compilation failed (Windows)**
```powershell
conda run -n fase3 pip install greenlet --only-binary :all:
```
Instale o greenlet primeiro, antes dos demais pacotes.

**Erro: "ModuleNotFoundError: No module named 'peft'"**
```powershell
pip install peft trl transformers accelerate
```

**CUDA nao detectado (GPU ignorada)**
```powershell
# Reinstale o PyTorch com suporte ao driver CUDA instalado
pip install torch --force-reinstall --index-url https://download.pytorch.org/whl/cu128
# Verifique:
python -c "import torch; print(torch.cuda.is_available())"
```

**Erro: "GEMINI_API_KEY not found" (aviso, nao erro critico)**
O sistema usa o modelo local como fallback automaticamente. Para usar o Gemini, crie o arquivo `.env` com sua chave.

**Erro ao baixar modelo do HuggingFace (sem internet)**
Os modelos sao baixados automaticamente na primeira execucao e ficam em cache. Sem internet, use `--demo-only` apos a primeira execucao.

**Fine-tuning muito lento**
O fine-tuning em CPU pode demorar muito. Use `--skip-finetune` para pular essa etapa na demonstracao. Para treinar com qualidade, use uma maquina com GPU.

---

## Configuracao da API do Gemini (Opcional)

1. Acesse: https://makersuite.google.com/app/apikey
2. Faca login com sua conta Google
3. Clique em "Create API Key"
4. Copie a chave gerada
5. Cole no arquivo `.env`:
   ```
   GEMINI_API_KEY=AIzaSy...sua_chave_aqui
   ```

O plano gratuito do Gemini e suficiente para todos os testes deste projeto.

---

## Autor

Matheus Tassi Souza - RM367424 | FIAP Pos-Tech IA para Devs
