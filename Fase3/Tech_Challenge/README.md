# Assistente Medico Virtual HospitalIQ

## Tech Challenge - Fase 3 | FIAP Pos-Tech IA para Devs

Bem-vindo ao meu projeto da Fase 3!

Depois de trabalhar com Machine Learning para diagnostico (Fase 1) e otimizacao de rotas com algoritmos geneticos e LLMs (Fase 2), chegou a hora de avançar para um sistema mais ambicioso: um assistente medico virtual treinado com dados proprios do hospital, capaz de auxiliar medicos e profissionais de saude com base nos protocolos internos.

O sistema integra tres tecnologias principais:
- Fine-tuning de LLM com LoRA para adaptar o modelo ao dominio medico
- LangChain com RAG (busca em base de conhecimento vetorial) para respostas contextualizadas
- LangGraph para orquestrar fluxos de decisao clinica automatizados

> Aviso importante: Este e um projeto academico. O assistente e uma ferramenta de apoio ao profissional de saude. A decisao clinica final e sempre responsabilidade do medico. O sistema nunca prescreve medicamentos diretamente.

---

## O que o sistema faz?

1. Recebe informacoes de um paciente (queixa, sinais vitais, comorbidades)
2. Classifica a gravidade (CRITICA / ALTA / MEDIA / BAIXA)
3. Sugere exames com base na queixa e na gravidade
4. Consulta os protocolos internos do hospital via busca semantica (RAG)
5. Gera uma resposta com condutas sugeridas, indicando a fonte de cada informacao
6. Emite alertas criticos quando necessario
7. Registra todo o fluxo em logs de auditoria

---

## Estrutura do Projeto

```
Fase3/Tech_Challenge/
├── data/
│   ├── pubmedqa/                       # Dataset PubMedQA original (pqal, folds 0-9 + test)
│   ├── pubmedqa_train.jsonl            # 450 exemplos de treino (fold 0, formato interno)
│   ├── pubmedqa_val.jsonl              # 50 exemplos de validacao (fold 0, formato interno)
│   ├── pubmedqa_test.jsonl             # 500 exemplos de teste (formato interno)
│   ├── medical_protocols.txt           # 6 protocolos clinicos do hospital (base RAG)
│   └── patient_records_sample.json     # 5 registros de pacientes sinteticos e anonimizados
├── notebooks/
│   └── 01_demonstracao_completa.ipynb  # Demonstracao interativa do sistema
├── results/
│   ├── data/                           # Splits gerados durante execucao
│   ├── modelos/                        # Modelo fine-tunado e base vetorial FAISS
│   ├── logs/                           # Logs de auditoria (JSONL diario)
│   └── graficos/                       # Visualizacoes geradas
├── src/
│   ├── __init__.py
│   ├── preprocessing.py        # Carregamento, anonimizacao e formatacao dos dados
│   ├── pubmedqa_converter.py   # Converte PubMedQA JSON -> JSONL interno
│   ├── fine_tuning.py          # Pipeline de fine-tuning com LoRA/PEFT + TRL
│   ├── assistant.py            # Assistente RAG com LangChain + Gemini
│   ├── langgraph_flows.py      # Grafo de fluxo clinico com LangGraph
│   ├── security.py             # Auditoria, validacao e seguranca
│   ├── evaluation.py           # BLEU, ROUGE e metricas de segurança
│   └── main.py                 # Ponto de entrada principal
├── tests/
│   └── test_assistant.py       # Testes unitarios (pytest)
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
| LLM base (fine-tuning demo) | google/flan-t5-base (HuggingFace) |
| Fine-tuning | PEFT (LoRA) + TRL (SFTTrainer) |
| LLM producao (assistente) | Google Gemini (com fallback automatico) |
| RAG Framework | LangChain (LCEL) |
| Busca vetorial | FAISS + sentence-transformers |
| Fluxo clinico | LangGraph StateGraph |
| Auditoria | Logging estruturado (JSONL) |
| Avaliacao | BLEU + ROUGE + Safety audit |
| Testes | pytest |

---

## Dataset

O projeto utiliza o **PubMedQA** (github.com/pubmedqa/pubmedqa), uma base publica de perguntas e respostas clinicas derivadas de publicacoes do PubMed, com respostas sim/nao/talvez e justificativas longas. Utilizamos o subset **pqal** (labeled), fold 0:

- **Treino**: 450 exemplos (`data/pubmedqa_train.jsonl`)
- **Validacao**: 50 exemplos (`data/pubmedqa_val.jsonl`)
- **Teste**: 500 exemplos (`data/pubmedqa_test.jsonl`)

Os dados foram convertidos para o formato interno do projeto via `src/pubmedqa_converter.py`. O modulo `preprocessing.py` aplica anonimizacao automatica (CPF, CNS, nomes, datas) antes do treinamento.

Alem do PubMedQA, o sistema inclui:
- `data/medical_protocols.txt` — 6 protocolos clinicos do hospital (base de conhecimento RAG)
- `data/patient_records_sample.json` — 5 registros de pacientes sinteticos e anonimizados

---

## Seguranca e Limites do Sistema

O assistente foi projetado com restricoes explicitas:

- Nunca prescreve medicamentos diretamente — qualquer resposta com prescricao direta e filtrada automaticamente
- Indica sempre a fonte dos protocolos utilizados na resposta
- Emite alertas criticos quando a gravidade for CRITICA
- Registra todas as consultas em logs de auditoria com hashes (sem armazenar dados do paciente)
- Valida as entradas do usuario contra tentativas de prompt injection
- Adiciona disclaimer de responsabilidade clinica em todas as respostas

---

## Autor

Matheus Tassi Souza - RM367424

FIAP Pos-Tech IA para Devs | Fase 3
