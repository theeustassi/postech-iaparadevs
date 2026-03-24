# Relatorio Tecnico - Tech Challenge Fase 3

## Assistente Medico Virtual HospitalIQ

**Autor:** Matheus Tassi Souza - RM367424
**Curso:** FIAP Pos-Tech IA para Devs
**Fase:** 3 | Data: 2025-2026

---

## 1. Resumo Executivo

Este projeto implementa um assistente medico virtual para o HospitalIQ, integrando tres componentes principais: um pipeline de fine-tuning de LLM com tecnica LoRA, um assistente baseado em RAG com LangChain + Google Gemini, e fluxos de decisao clinica automatizados com LangGraph.

O sistema e capaz de classificar a gravidade de um paciente, consultar protocolos internos via busca semantica e gerar sugestoes de conduta com fonte identificada - tudo com controles de seguranca que impedem prescricoes diretas e registram cada acao em logs de auditoria.

---

## 2. Dataset

### 2.1 Origem dos Dados

O projeto utiliza o **PubMedQA** (https://github.com/pubmedqa/pubmedqa), uma base publica de perguntas e respostas clinicas derivadas de publicacoes do PubMed. Utilizamos o subset **pqal** (labeled), que contem 1000 exemplos com respostas binarizadas (yes/no/maybe) e justificativas longas extraidas de artigos cientificos.

Os dados originais do PubMedQA foram convertidos para o formato interno do projeto pelo modulo `src/pubmedqa_converter.py`, mapeando:
- `QUESTION` → `question`
- `CONTEXTS` + `LABELS` → `context` (contexto anotado com relevancia)
- `LONG_ANSWER` + `final_decision` → `answer`
- Metadados: `pmid`, `year`, `final_decision`, `meshes`

### 2.2 Formato do Dataset

Cada exemplo no arquivo `data/pubmedqa_train.jsonl` segue a estrutura:

```json
{
  "id": "11011694",
  "question": "Is laparoscopic conversion during adrenalectomy more frequent with larger tumors?",
  "context": "[RELEVANT] Current evidence suggests ...",
  "answer": "yes. Laparoscopic conversion was more frequent...",
  "type": "pubmedqa",
  "source": "PubMedQA-pqal-fold0",
  "metadata": {"pmid": "11011694", "year": 2000, "final_decision": "yes", "meshes": [...]}
}
```

### 2.3 Estatisticas do Dataset

| Metrica | Valor |
|---|---|
| Total de exemplos (pqal) | 1000 |
| Split treino (fold 0) | 450 |
| Split validacao (fold 0) | 50 |
| Split teste | 500 |
| Distribuicao yes/no/maybe | ~55% / ~28% / ~17% |
| Topicos cobertos | Multiplas especialidades clinicas e cirurgicas |

### 2.4 Anonimizacao

O modulo `preprocessing.py` aplica as seguintes tecnicas de anonimizacao:
- Substituicao de CPF, CNPJ, CNS e RG por marcadores
- Hash SHA-256 truncado nos IDs de pacientes
- Remocao de nomes proprios em tres palavras
- Substituicao de datas por marcadores
- Remocao de telefones e enderecos

---

## 3. Fine-Tuning do LLM

### 3.1 Modelo Base

O projeto suporta dois modos de execucao com modelos diferentes:

- **Pipeline CLI (`main.py`)**: usa `google/flan-t5-base` (250M params, seq2seq) com `max_steps=50` — roda em CPU, ideal para validar o fluxo completo rapidamente.
- **Notebook de demonstracao (`01_demonstracao_completa.ipynb`)**: usa `microsoft/Phi-3-mini-4k-instruct` (3.8B params, causal decoder-only) com treinamento completo (3 epocas) — requer GPU, e a configuracao de referencia para avaliacao de qualidade.

| Modelo | Tamanho | Req. de Hardware | Onde e usado |
|---|---|---|---|
| google/flan-t5-base | 250M params | CPU/GPU | **`main.py` (demo rapido, padrao CLI)** |
| microsoft/Phi-3-mini-4k-instruct | 3.8B params | GPU 8GB+ fp16 | **Notebook (treinamento completo, GPU)** |
| BioMistral/BioMistral-7B | 7B params | GPU 6GB+ QLoRA | Alternativa medica especializada |
| meta-llama/Llama-3.2-3B-Instruct | 3B params | GPU 8GB+ fp16 | Alternativa custo-beneficio |

### 3.2 Tecnica: LoRA (Low-Rank Adaptation)

O LoRA e uma tecnica de Parameter-Efficient Fine-Tuning (PEFT) que:

1. Congela todos os pesos originais do modelo
2. Insere matrizes de baixo rank (rank r) nas camadas de atencao
3. Treina apenas essas matrizes adicionais

A formula matematica da adaptacao LoRA e:

```
W' = W + delta_W = W + B * A
```

Onde W e o peso original congelado, A e do tamanho (r x d_in) e B e do tamanho (d_out x r), com r << d_in.

### 3.3 Configuracoes do Fine-Tuning

Existem duas configuracoes em uso no projeto:

| Parametro | `main.py` (demo CLI) | Notebook (GPU, referencia) |
|---|---|---|
| Modelo | google/flan-t5-base | microsoft/Phi-3-mini-4k-instruct |
| LoRA rank (r) | 8 | 16 |
| LoRA alpha | 32 | 32 |
| LoRA dropout | 0.05 | 0.05 |
| Target modules | detectado automaticamente | q_proj, v_proj |
| Learning rate | 2e-4 | 2e-4 |
| Batch size | 2 | 4 |
| Gradient accumulation | 2 steps (batch efetivo 4) | 4 steps (batch efetivo 16) |
| max_steps | 50 (demo rapido) | -1 (usa num_train_epochs) |
| Epocas | — | 3 (dataset completo, 450 exemplos) |
| Max seq length | 512 tokens | 512 tokens |
| QLoRA (4-bit) | False | False (opcional para 7B+) |
| Hardware | CPU/GPU | RTX 5070 Ti (16GB), fp16 |

Com o `flan-t5-base`, o LoRA reduz os parametros treinaveis de 248M para aproximadamente 1.2M (menos de 0.5%), tornando o fine-tuning viavel em CPUs durante a demonstracao CLI. No notebook, o Phi-3-mini-4k-instruct e treinado com epocas completas em GPU para qualidade de referencia.

### 3.4 Formato de Treinamento

Os dados sao formatados no padrao instruction-following:

```
### Instrucao:
Voce e um assistente medico especializado...

### Contexto:
[trecho do protocolo hospitalar relevante]

### Pergunta:
[pergunta medica do usuario]

### Resposta:
[resposta esperada baseada no protocolo]
```

Esse formato e amplamente utilizado em modelos instruction-tuned (como o Alpaca e o LLaMA-2-Chat) e permite que o modelo aprenda a seguir instrucoes medicas especificas.

---

## 4. Assistente Medico com LangChain

### 4.1 Arquitetura RAG

O assistente utiliza RAG (Retrieval-Augmented Generation) para responder perguntas com base nos documentos do hospital, sem memorizar tudo no modelo.

```
Pergunta do profissional de saude
    |
    v
Embeddings (sentence-transformers/all-MiniLM-L6-v2)
    |
    v
FAISS VectorStore  <--- Base de conhecimento
(top-3 chunks mais  (protocols.txt + synthetic_qa.jsonl)
 relevantes)
    |
    v
Prompt formatado (SYSTEM + contexto + pergunta)
    |
    v
LLM (Google Gemini ou modelo local)
    |
    v
Resposta com fonte + disclaimer de segurança
```

### 4.2 Diagrama do Fluxo LangChain (LCEL)

```
rag_chain = (
    {
        "context": retriever | format_context_with_sources,
        "question": RunnablePassthrough()
    }
    | ChatPromptTemplate([system_prompt, human_message])
    | ChatGoogleGenerativeAI (Gemini)
    | StrOutputParser()
)
```

O LCEL (LangChain Expression Language) permite compor o pipeline de forma declarativa e eficiente, com suporte a streaming e paralelismo quando necessario.

### 4.3 Estrategia de Embeddings

Utilizamos o modelo `sentence-transformers/all-MiniLM-L6-v2` para gerar os embeddings dos documentos:

- Modelo de 90 MB, roda em CPU
- Gera vetores de 384 dimensoes
- Treinado para similaridade semantica em multiplos idiomas
- Nao requer API key

Os documentos sao divididos em chunks de 600 tokens com sobreposicao de 60 tokens para manter contexto nas bordas.

### 4.4 LLM: Google Gemini com Fallback

O sistema tenta conectar ao Gemini na ordem: `gemini-2.5-flash-lite`, `gemini-2.5-flash`, `gemini-1.5-flash`, `gemini-2.0-flash`, `gemini-1.5-pro`. Se nenhum estiver disponivel (sem API key ou sem conexao), carrega o modelo local fine-tunado (ou flan-t5-base).

---

## 5. Fluxos Clinicos com LangGraph

### 5.1 Conceito

O LangGraph e uma extensao do LangChain para criar fluxos multi-etapas com estado persistente. Diferente de um simples pipeline, o LangGraph permite:
- Ramificacao condicional (ex: rota diferente para caso CRITICO)
- Acumulacao de estado ao longo do fluxo
- Registro de auditoria em cada etapa

### 5.2 Diagrama do Grafo Clinico

```
[INICIO]
    |
patient_intake
(recebe e normaliza dados do paciente)
    |
triage
(classifica gravidade: LOW / MEDIUM / HIGH / CRITICAL)
    |
check_exams
(verifica e sugere exames pendentes)
    |
    +------ [severity == CRITICAL] ------> critical_alert
    |                                          |
    +------ [severity != CRITICAL] -----> clinical_reasoning
                                               |
                                   (consulta assistente RAG)
                                               |
    <------------------------------------------+
    |
safety_check
(filtra prescricoes diretas, adiciona disclaimers)
    |
generate_response
(formata resposta final para o profissional)
    |
audit_logging
(registra todo o fluxo para rastreabilidade)
    |
[FIM]
```

### 5.3 Estado do Grafo (PatientState)

```python
class PatientState(TypedDict):
    patient_id: str
    patient_info: dict
    query: str
    severity: str              # LOW | MEDIUM | HIGH | CRITICAL
    severity_reason: str
    pending_exams: List[str]
    clinical_suggestions: str
    critical_alerts: List[str]
    safety_check_passed: bool
    final_response: str
    audit_trail: Annotated[List[dict], operator.add]  # acumulativo
    error: Optional[str]
```

O campo `audit_trail` usa `operator.add` como redutor, o que faz o LangGraph acumular as entradas de auditoria de cada no em vez de substitui-las.

### 5.4 Criterios de Triagem

A classificacao de gravidade segue diretrizes baseadas no Protocolo de Manchester:

| Gravidade | Exemplos de criterios |
|---|---|
| CRITICAL | SpO2 < 90%, PAS < 80 mmHg, FC > 150, palavras-chave de emergencia |
| HIGH | SpO2 < 94%, PAS < 95 mmHg, queixas de urgencia (dor toracica, dispneia) |
| MEDIUM | Temperatura > 37.8 C, FC > 100, sintomas moderados |
| LOW | Sinais vitais normais, sem sinais de alarme |

---

## 6. Seguranca e Validacao

### 6.1 Controles de Seguranca Implementados

1. **Proibicao de prescricao direta**: O `safety_check_node` filtra qualquer resposta que contenha termos de prescricao direta ("tome ", "use ", "administre ", "mg/dia", etc.)

2. **Disclaimer obrigatorio**: Toda resposta inclui aviso de que a decisao clinica final e responsabilidade do medico assistente.

3. **Validacao de entrada**: O `InputValidator` verifica:
   - Comprimento maximo da pergunta (2000 caracteres)
   - Tentativas de prompt injection (10+ padroes detectados)
   - Sanitizacao de caracteres de controle

4. **Logging de auditoria**: Toda consulta, alerta e bloqueio de seguranca e registrado em arquivo JSONL diario com:
   - IDs de pacientes mascarados
   - Hashes SHA-256 dos conteudos (sem armazenar texto sensivel)
   - Timestamp UTC
   - Gravidade e fontes consultadas

5. **Anonimizacao de dados**: O preprocessador remove automaticamente CPF, RG, CNS, nomes completos e datas dos dados antes do treinamento.

### 6.2 Explainability

Cada resposta do assistente inclui a lista de fontes consultadas pelo retriever:
```
Fontes: data/medical_protocols.txt, Protocolo HospitalIA v2.0 - Sepse
```

Isso permite que o profissional de saude verifique a origem da informacao e avalie sua confiabilidade.

---

## 7. Avaliacao do Modelo

### 7.1 Metricas Utilizadas

| Metrica | O que mede | Como interpretar |
|---|---|---|
| BLEU | Sobreposicao de n-gramas com a referencia | 0 = nenhuma sobreposicao, 1 = perfeito |
| ROUGE-1 | Sobreposicao de unigramas | Acima de 0.4 e considerado bom |
| ROUGE-2 | Sobreposicao de bigramas | Mais restritivo que ROUGE-1 |
| ROUGE-L | Maior subsequencia comum | Mais robusto a reformulacoes |
| Exact Match | % de respostas identicas (normalizado) | Geralmente baixo em perguntas abertas |
| Safety Compliance | % de respostas sem prescricao direta | Meta: 100% |
| Disclaimer Rate | % de respostas com aviso de responsabilidade | Meta: 100% |

### 7.2 Resultado Esperado (modelo demo)

O `flan-t5-base` com treinamento completo no PubMedQA (450 exemplos, fold 0) produz metricas melhores do que a configuracao de demo anterior. O objetivo e mostrar o pipeline funcionando end-to-end com dados reais: que o modelo aprende o formato de instrucao medica e que as metricas de seguranca atingem 100%.

O treinamento foi realizado com suporte a GPU (NVIDIA RTX 5070 Ti, 16 GB VRAM, fp16 habilitado). Para metricas clinicamente relevantes, seria necessario:
- Modelo maior (BioMistral-7B ou LLaMA-2-7b com QLoRA)
- Treinamento de 3-5 epocas completas com learning rate scheduling

### 7.3 Avaliacao do Assistente RAG

O assistente RAG (Gemini + FAISS) tende a ter qualidade muito superior ao modelo fine-tunado em CPU, pois o Gemini ja e um modelo grande e o RAG garante que as respostas sejam baseadas nos protocolos reais. Os controles de seguranca (disclaimer rate = 100%, safety compliance = 100%) sao garantidos pelo `safety_check_node` independentemente da qualidade do LLM.

---

## 8. Discussao e Limitacoes

**Pontos fortes:**
- Pipeline modular e bem separado (preprocessing, fine-tuning, assistant, security, evaluation)
- Seguranca implementada em multiplas camadas (validacao de entrada, filtro de output, logging)
- Treinamento com dataset real (PubMedQA, 450 exemplos clinicos de publicacoes do PubMed)
- GPU utilizada na fase de fine-tuning e embeddings (RTX 5070 Ti, fp16)
- Fallback automatico entre modelos Gemini e para modelo local

**Limitacoes:**
- O `flan-t5-base` e um modelo seq2seq pequeno (250M params); qualidade clinica real exigiria um modelo medico especializado (BioMistral-7B, LLaMA-2-7b)
- A triagem por palavras-chave e simplificada; um sistema real usaria NLP mais robusto (NER medico)
- Sem integracao com sistemas de prontuario eletronico (EHR) reais
- Apenas fold 0 do PubMedQA e utilizado; cross-validation nos 10 folds aumentaria a confiabilidade

---

## 9. Referencias

- PubMedQA: https://github.com/pubmedqa/pubmedqa
- HuggingFace PEFT: https://github.com/huggingface/peft
- TRL (SFTTrainer): https://github.com/huggingface/trl
- LangChain Docs: https://python.langchain.com/docs
- LangGraph Docs: https://langchain-ai.github.io/langgraph
- Google Gemini API: https://makersuite.google.com/app/apikey
