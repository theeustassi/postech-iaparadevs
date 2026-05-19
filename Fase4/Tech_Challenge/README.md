# Sistema de Monitoramento Multimodal para Saude da Mulher - HospitalIQ

## Tech Challenge - Fase 4 | FIAP Pos-Tech IA para Devs

Este projeto implementa um sistema de monitoramento continuo de pacientes por meio de dados
multimodais (video, audio e texto) para identificar sinais precoces de risco especificos da
saude e seguranca feminina.

O sistema e uma evolucao direta do assistente medico da Fase 3: enquanto aquele processava
laudos e exames textuais com LLM+RAG, este adiciona duas novas modalidades de entrada —
video clinico e audio de consultas — combinando os resultados em uma avaliacao de risco
unificada por fusao multimodal.

---

## O que o sistema faz?

1. Recebe um video clinico (cirurgia, consulta, fisioterapia ou triagem) e/ou um audio de consulta medica
2. Processa o video frame a frame, detectando sangramento anomalo (cirurgia) ou padroes de linguagem corporal (consultas)
3. Extrai features acusticas do audio e classifica indicadores de depressao pos-parto, ansiedade gestacional, fadiga hormonal e violencia domestica
4. Combina os resultados das duas modalidades por fusao multimodal ponderada
5. Classifica o risco em quatro niveis de prioridade: VERDE, AMARELO, LARANJA e VERMELHO
6. Gera relatorio clinico automatizado com alertas e recomendacoes de conduta
7. Registra todo o fluxo em logs de auditoria anonimizados (conformidade LGPD)

---

## Estrutura do Projeto

```
Fase4/Tech_Challenge/
├── data/
│   ├── amostras_video/             # Pasta para videos de demonstracao (.mp4)
│   └── amostras_audio/             # Pasta para audios de demonstracao (.wav)
├── notebooks/
│   └── 01_demonstracao_completa.ipynb  # Pipeline interativo completo
├── results/
│   ├── graficos/                   # Graficos gerados automaticamente
│   ├── relatorios/                 # Relatorios JSON por paciente (anonimizados)
│   └── modelos/                    # Pesos de modelos (YOLOv8, etc.)
├── src/
│   ├── __init__.py
│   ├── video_analysis.py           # Pipeline de analise de video (YOLOv8 + OpenCV)
│   ├── audio_analysis.py           # Pipeline de analise de audio (librosa + Whisper)
│   ├── multimodal_fusion.py        # Fusao ponderada e classificacao de prioridade
│   ├── report_generator.py         # Geracao de relatorios clinicos
│   ├── security.py                 # Anonimizacao, auditoria e validacao de entradas
│   ├── visualization.py            # Graficos matplotlib
│   └── main.py                     # Ponto de entrada (CLI)
├── tests/
│   └── test_pipeline.py            # Testes unitarios (pytest)
├── Dockerfile
├── requirements.txt
├── .env.example
├── README.md
├── RELATORIO_TECNICO.md
└── GUIA_EXECUCAO.md
```

---

## Stack de Tecnologias

| Componente | Tecnologia |
|---|---|
| Deteccao de objetos em video | YOLOv8 (ultralytics) |
| Estimativa de pose | YOLOv8-pose |
| Processamento de frames | OpenCV |
| Features acusticas | librosa |
| Transcricao de audio | OpenAI Whisper (small, local) |
| Classificacao de emocoes | Transformers (HuggingFace) |
| Fusao multimodal | combinacao linear ponderada |
| Relatorios e logs | JSON / JSONL |
| Visualizacoes | matplotlib / seaborn |
| Testes | pytest |
| Containerizacao | Docker |

---

## Objetivos Contemplados

- Detectar precocemente riscos em saude materna e ginecologica (analise de video cirurgico)
- Identificar sinais de violencia domestica ou abuso (audio + linguagem corporal)
- Monitorar bem-estar psicologico feminino (depressao pos-parto, ansiedade gestacional)

## Funcionalidades Implementadas

- Analise de videos de cirurgias ginecologicas com deteccao de sangramento anomalo
- Processamento de gravacoes de voz detectando depressao pos-parto, ansiedade e violencia domestica

---

## Como Executar

Consulte o [GUIA_EXECUCAO.md](GUIA_EXECUCAO.md) para instrucoes detalhadas.

```powershell
# Rapido: demonstracao com dados sinteticos
cd "postech-iaparadevs\Fase4\Tech_Challenge"
conda create -n fase4 python=3.11 -y
conda activate fase4
pip install -r requirements.txt
python src/main.py
```

---

## Aviso Legal

Este e um projeto academico de pos-graduacao. O sistema e uma ferramenta de apoio ao
profissional de saude. A decisao clinica final e sempre responsabilidade do medico.
O sistema nunca substitui avaliacao medica presencial.

Todos os dados utilizados nos testes sao sinteticos. Nenhum dado real de paciente
foi coletado, armazenado ou processado neste projeto.
