# Guia de Execucao - Tech Challenge Fase 4

## Sistema de Monitoramento Multimodal para Saude da Mulher

---

## Pre-requisitos

- Python 3.11+
- Conda (Anaconda ou Miniconda) — recomendado
- 4 GB de RAM (minimo para operacao sem Whisper e sem YOLOv8)
- 8 GB de RAM (recomendado para Whisper small + YOLOv8)
- GPU NVIDIA com CUDA 12.x (opcional — acelera Whisper e YOLOv8 significativamente)
- Conexao com a internet (para baixar modelos do HuggingFace e ultralytics na primeira execucao)
- Webcam ou arquivo de video/audio para analise com dados reais (opcional)

---

## Opcao 1: Script Principal (Recomendado)

Tempo estimado: 3-5 minutos na primeira execucao (baixa pesos do YOLOv8 e Whisper quando disponiveis)

```powershell
# 1. Acesse o diretorio do projeto
cd "postech-iaparadevs\Fase4\Tech_Challenge"

# 2. Crie e ative o ambiente conda
conda create -n fase4 python=3.11 -y
conda activate fase4

# 3. Instale as dependencias
pip install -r requirements.txt

# 4. (Opcional, GPU NVIDIA) Reinstale o PyTorch com suporte CUDA 12.8
pip install torch torchvision torchaudio --force-reinstall --index-url https://download.pytorch.org/whl/cu128

# 5. Execute o pipeline completo com dados sinteticos
python src/main.py
```

### Modos de execucao

```powershell
# Demonstracao completa com todos os cenarios sinteticos (padrao)
python src/main.py

# Apenas video cirurgico sintetico
python src/main.py --tipo cirurgia

# Apenas pos-parto sintetico
python src/main.py --tipo consulta

# Analisar arquivos reais
python src/main.py --video data/amostras_video/cirurgia.mp4 --audio data/amostras_audio/consulta.wav --tipo cirurgia

# Analisar video real com tipo de audio diferente
python src/main.py --video data/amostras_video/triagem.mp4 --audio data/amostras_audio/depoimento.wav --tipo triagem --tipo-audio triagem --paciente-id PAC-001

# Apenas audio (sem video)
python src/main.py --audio data/amostras_audio/pos_parto.wav --tipo pos_parto --tipo-audio pos_parto
```

### Argumentos disponíveis

| Argumento | Descricao | Valores |
|---|---|---|
| `--video` | Caminho para arquivo de video | qualquer .mp4, .avi, .mov |
| `--audio` | Caminho para arquivo de audio | qualquer .wav (16kHz recomendado) |
| `--tipo` | Tipo de video/atendimento | cirurgia, consulta, fisioterapia, triagem |
| `--tipo-audio` | Tipo de consulta para o audio | ginecologica, pre_natal, pos_parto, triagem |
| `--paciente-id` | ID da paciente (sera anonimizado) | qualquer string alfanumerica |

### Arquivos gerados apos a execucao

```
results/
├── graficos/
│   ├── risco_video_cirurgia.png          # Curva de risco por frame
│   ├── pontuacoes_audio_pos_parto.png    # Indicadores por consulta
│   └── fusao_multimodal_comparativo.png  # Painel comparativo
├── relatorios/
│   └── relatorio_<hash>_<timestamp>.json # Relatorio por paciente (anonimizado)
└── logs/
    └── audit_YYYYMMDD.jsonl              # Log de auditoria do dia
```

---

## Opcao 2: Jupyter Notebook Interativo

```powershell
# No ambiente conda ja ativado com as dependencias instaladas:
cd "postech-iaparadevs\Fase4\Tech_Challenge"
jupyter notebook notebooks/01_demonstracao_completa.ipynb
```

O notebook demonstra cada etapa do pipeline de forma interativa:
- Secao 1: Analise de video (4 tipos de cenario)
- Secao 2: Analise de audio (4 tipos de consulta)
- Secao 3: Fusao multimodal e relatorios
- Secao 4: Seguranca, anonimizacao e validacao de entradas
- Secao 5: Tabela de resumo dos resultados

---

## Opcao 3: Docker

```powershell
# Build da imagem
cd "postech-iaparadevs\Fase4\Tech_Challenge"
docker build -t hospitaliq-fase4 .

# Executar demonstracao sintetica
docker run --rm hospitaliq-fase4

# Executar com arquivo de video real (montar volume)
docker run --rm -v "${PWD}/data:/app/data" -v "${PWD}/results:/app/results" hospitaliq-fase4 python src/main.py --video data/amostras_video/cirurgia.mp4 --tipo cirurgia
```

---

## Executar os Testes

```powershell
# No ambiente conda ativado
cd "postech-iaparadevs\Fase4\Tech_Challenge"
pytest tests/ -v

# Com cobertura de codigo
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## Dependencias Opcionais e Seus Efeitos

| Dependencia | Instalada | Nao instalada |
|---|---|---|
| `ultralytics` (YOLOv8) | Pose estimation e deteccao de objetos precisas | Fallback para fluxo optico (menor precisao) |
| `librosa` | Features acusticas de alta qualidade (MFCC, pitch, onset) | Features basicas via numpy |
| `openai-whisper` | Transcricao de audio para texto | Sem transcricao (so features acusticas) |
| `transformers` | Analise de sentimento via modelo pre-treinado | Analise apenas por regras linguisticas |

O sistema funciona sem nenhuma dessas dependencias opcionais, usando as implementacoes de fallback.
Para maxima precisao clinica, recomenda-se instalar todas.

