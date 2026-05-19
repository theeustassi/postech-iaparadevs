# Relatorio Tecnico - Tech Challenge Fase 4

## Sistema de Monitoramento Multimodal para Saude da Mulher

**Autor:** Matheus Tassi Souza - RM367424
**Curso:** FIAP Pos-Tech IA para Devs
**Fase:** 4 | Data: 2026

---

## 1. Resumo Executivo

Este projeto implementa um sistema de monitoramento multimodal para saude da mulher que
processa simultaneamente video clinico e audio de consultas medicas, combinando os resultados
por fusao ponderada para gerar alertas clinicos priorizados.

O sistema contempla tres dos objetivos propostos pelo desafio:

1. Detectar precocemente riscos em saude materna e ginecologica — via analise de video cirurgico com
   deteccao de sangramento anomalo usando YOLOv8 + mascara de cor HSV.
2. Identificar sinais de violencia domestica ou abuso — via analise de audio (features acusticas + Whisper)
   e analise de linguagem corporal em consultas de triagem.
3. Monitorar bem-estar psicologico feminino — via classificadores de depressao pos-parto,
   ansiedade gestacional e fadiga hormonal baseados em prosodia vocal.

Duas das funcionalidades opcionais foram implementadas:
- Analise de videos de cirurgias ginecologicas, consultas, fisioterapia e triagem de violencia
- Processamento de gravacoes de voz com deteccao de depressao pos-parto, ansiedade e violencia domestica

---

## 2. Arquitetura do Sistema

### 2.1 Fluxo Multimodal

```
             ┌─────────────┐      ┌──────────────────────┐
             │  Video MP4  │─────>│  AnalisadorVideo      │
             └─────────────┘      │  (video_analysis.py)  │
                                  │  - DetectorSangramento│
                                  │  - AnalisadorPostura  │     ┌─────────────────┐
                                  │  - AnalisadorFisio    │────>│                 │
                                  └──────────────────────┘     │  FusorMultimodal│
                                                                │  (combinacao    │
             ┌─────────────┐      ┌──────────────────────┐     │   ponderada)    │
             │  Audio WAV  │─────>│  AnalisadorAudio      │────>│                 │
             └─────────────┘      │  (audio_analysis.py)  │     └────────┬────────┘
                                  │  - ExtratorFeatures   │              │
                                  │  - TranscritorWhisper │    ┌─────────v────────┐
                                  │  - Clf.Depressao      │    │  GeradorRelatorio│
                                  │  - Clf.Ansiedade      │    │  + AuditLogger   │
                                  │  - Clf.Violencia      │    └──────────────────┘
                                  └──────────────────────┘
```

### 2.2 Modulos

| Modulo | Responsabilidade |
|---|---|
| `video_analysis.py` | Processamento de frames, deteccao de sangramento, analise de postura e movimento |
| `audio_analysis.py` | Extracao de features acusticas, transcricao e classificacao de risco vocal |
| `multimodal_fusion.py` | Fusao ponderada, classificacao de prioridade e recomendacoes de conduta |
| `report_generator.py` | Geracao de relatorios JSON anonimizados e saida textual formatada |
| `security.py` | Anonimizacao LGPD, audit log JSONL, validacao de entradas (anti-injection) |
| `visualization.py` | Graficos de risco por frame, indicadores de audio e comparativo multimodal |
| `main.py` | CLI com suporte a arquivos reais e modo demo sintetico |

---

## 3. Modulo de Video

### 3.1 Deteccao de Sangramento Anomalo (Cirurgia)

A deteccao de sangramento utiliza analise de cor no espaco HSV, uma tecnica estabelecida para
segmentacao de sangue em imagens cirurgicas endoscopicas e laparoscopicas.

**Estrategia implementada:**
- Conversao do frame BGR para HSV
- Aplicacao de duas mascaras para tons de vermelho escuro tipicos de sangue: H∈[0,10] e H∈[160,180], S>120, V>70
- Calculo da proporcao de pixels dentro da mascara em relacao a area total do frame
- Classificacao por limiares: proporcao >= 8% = ALERTA, proporcao >= 20% = CRITICO

**Limitacao e evolucao para producao:**
Em producao, esta etapa seria substituida por um modelo YOLOv8 customizado treinado em imagens
cirurgicas laparoscopicas anotadas (ex: dataset Cholec80 ou CholecT50 para identificacao de
instrumentos e tecidos). A analise de cor e suficiente como prova de conceito, mas tem falsos
positivos com instrumentos cirurgicos de cor vermelha ou iluminacao inadequada.

### 3.2 Analise de Linguagem Corporal (Consultas e Triagem)

Para consultas e triagem de violencia, o modulo usa YOLOv8-pose para estimativa de 17 keypoints
COCO por pessoa presente no frame.

**Indicadores monitorados:**
- Postura fechada: pulsos cruzados proximos a linha central do corpo (indicador de defesa ou desconforto)
- Cabeca inclinada: altura do nariz abaixo de 80% da altura media dos ombros (esquivamento de contato visual)

Quando YOLOv8 nao esta disponivel, o modulo opera em modo fallback usando diferenca de frames
para detectar movimentos bruscos (agitacao, tremor).

### 3.3 Analise de Movimento em Fisioterapia

Para sessoes de fisioterapia, o modulo calcula a assimetria bilateral dos membros superiores
comparando as amplitudes verticais dos pulsos em relacao aos ombros correspondentes. Assimetria
acima de 40% indica possivel compensacao por dor, gerando alerta para o fisioterapeuta.

---

## 4. Modulo de Audio

### 4.1 Extracao de Features Acusticas

O modulo extrai as seguintes features usando librosa (quando disponivel):

| Feature | Descricao | Relevancia clinica |
|---|---|---|
| Energia RMS | Amplitude media do sinal | Hipofonia em depressao |
| Pitch medio (F0) | Frequencia fundamental media | Nivel de arousal emocional |
| Desvio de pitch | Variabilidade da F0 | Instabilidade emocional (ansiedade) |
| Taxa de pausa | Proporcao do audio em silencio | Retardo psicomotor (depressao) |
| Pausa maxima | Duracao da pausa mais longa | Hesitacao por medo (violencia) |
| Taxa de fala | Estimativa de silabas/segundo | Agitacao (ansiedade) ou lentidao (depressao) |
| MFCC (13 coef.) | Forma espectral da voz | Identidade vocoide e qualidade vocal |

### 4.2 Transcricao com Whisper

O sistema usa o modelo `whisper-small` da OpenAI, executado integralmente de forma local,
sem envio de dados para servicos externos. O modelo foi escolhido pelo equilibrio entre
precisao em portugues brasileiro e consumo de memoria (aproximadamente 500 MB).

### 4.3 Classificadores de Risco

**Depressao Pos-Parto:**
Baseado nas caracteristicas prosodicas de depressao descritas na literatura de speech pathology:
energia vocal reduzida (pondera 30%), alta taxa de pausa (20%), pausa maxima longa (15%),
fala lenta (15%) e prosodia monotona (10%). Vocabulario depressivo na transcricao adiciona
ate 30% ao score. Escore total acima de 0.4 recomenda aplicacao formal da Escala de Edimburgo (EPDS).

**Ansiedade Gestacional:**
Variacao de pitch elevada (30%), fala acelerada (25%) e fala entrecortada com alta energia (20%).
Vocabulario ansioso na transcricao adiciona ate 30%.

**Violencia Domestica:**
O relato direto de violencia (frases como "ele me bate", "tenho medo dele") tem peso de 70% do score.
Adicionalmente: voz extremamente baixa (20%), pausas longas pre-resposta (15%) e linguagem evasiva
("nao foi nada", "foi sem querer") acrescentam ate 25%. Score acima de 0.4 aciona recomendacao de
encaminhamento para protocolo de violencia domestica (Lei Maria da Penha).

**Fadiga Hormonal:**
Energia muito baixa (40%), fala muito lenta (30%) e relato verbal de cansaco (30%).

### 4.4 Pesos por Tipo de Consulta

Os classificadores sao combinados por soma ponderada de acordo com o tipo de consulta:

| Tipo | Depressao | Ansiedade | Violencia | Fadiga |
|---|---|---|---|---|
| Ginecologica | 20% | 30% | 20% | 30% |
| Pre-natal | 30% | 50% | 10% | 10% |
| Pos-parto | 50% | 20% | 10% | 20% |
| Triagem violencia | 10% | 10% | 70% | 10% |

---

## 5. Fusao Multimodal

### 5.1 Estrategia de Fusao

A fusao utiliza combinacao linear ponderada (late fusion), onde cada modalidade contribui
com um peso configuravel por tipo de atendimento:

```
pontuacao_fusao = peso_video * pontuacao_video + peso_audio * pontuacao_audio
```

A late fusion foi preferida sobre early fusion (concatenacao de features) ou decision fusion
(votacao) por tres razoes:

1. Permite que cada modalidade seja treinada e avaliada independentemente
2. Facilita o modo degradado quando uma modalidade nao esta disponivel
3. Os pesos podem ser ajustados por profissional clinico sem re-treinamento de modelos

### 5.2 Niveis de Prioridade

| Score de fusao | Nivel | Acao recomendada |
|---|---|---|
| < 0.20 | VERDE | Acompanhamento ambulatorial regular |
| 0.20 - 0.44 | AMARELO | Revisao medica em ate 48 horas |
| 0.45 - 0.69 | LARANJA | Acionar medico de plantao em ate 24 horas |
| >= 0.70 | VERMELHO | Intervencao imediata |

---

## 6. Resultados Obtidos

### 6.1 Cenarios de Demonstracao

Os testes foram realizados com dados sinteticos controlados, permitindo validar
o pipeline sem dados reais de pacientes:

**Cenario 1 - Cirurgia ginecologica sintetica:**
- Frame 0-15: fundo verde uniforme (tecido sem sangramento) -> risco < 0.01
- Frame 16-30: mancha vermelha crescente simulando sangramento -> risco escalando de 0.05 a 0.95
- Score de video: 0.35 (media); 0.95 (maximo)
- Alertas criticos: "CRITICO: sangramento anomalo extenso detectado (X% da area)"

**Cenario 2 - Consulta pos-parto sintetica:**
- Audio com energia 0.10 (baixa), taxa de pausa 0.60, fala lenta (0.8 sil/s)
- Transcricao: "estou muito cansada, fico chorando sem motivo, me sinto sozinha"
- Score de depressao: 0.75; Score de fusao (peso audio 75%): 0.57
- Nivel: LARANJA

**Cenario 3 - Triagem de violencia sintetica:**
- Audio com energia 0.05 (muito baixa), pausa maxima 4.0s
- Transcricao: "nao foi nada, estou bem, ele e bom, foi sem querer, por favor nao conte para ninguem"
- Score de violencia: 0.45; Score de fusao (peso audio 65%): 0.31-0.45
- Nivel: AMARELO a LARANJA (dependendo do score de video)
- Alertas: linguagem evasiva, voz extremamente baixa, pausa prolongada

**Cenario 4 - Acompanhamento pre-natal sintetico:**
- Audio com pitch modulado (desvio alto, simulando ansiedade)
- Score de ansiedade: 0.35; Score de fusao: 0.25
- Nivel: AMARELO

### 6.2 Validacao dos Classificadores

Os testes unitarios (`tests/test_pipeline.py`) cobrem 40 casos de teste, incluindo:
- Deteccao correta de sangramento em frames com mancha vermelha > 8% da area
- Alta pontuacao de depressao com features de energia baixa + pausas longas
- Alerta critico com relato direto de violencia na transcricao
- Pontuacoes sempre no intervalo [0, 1]
- Anonimizacao correta de CPF e IDs nos logs
- Rejeicao de tentativas de prompt injection e SQL injection

---

## 7. Seguranca e Conformidade LGPD

O sistema implementa os seguintes controles para protecao de dados de saude:

**Anonimizacao automatica:**
- CPF, CNS, RG, CNPJ substituidos por marcadores antes de qualquer log
- Datas de nascimento substituidas por [DATA-ANONIMIZADA]
- Nomes proprios (tres ou mais palavras) substituidos por [NOME-ANONIMIZADO]
- Telefones e enderecos removidos

**Controle de acesso aos logs:**
- IDs de pacientes mascarados nos logs (preserva primeiros 2 e ultimos 2 caracteres)
- Hashes SHA-256 truncados nos campos de query para rastreabilidade sem exposicao de dados
- Logs em formato JSONL diario, permitindo rotacao e auditoria independente

**Minimizacao de dados:**
- Transcricoes de audio nao sao persistidas em disco; apenas metricas derivadas
- Nenhum frame de video e armazenado; apenas pontuacoes de risco por timestamp
- Relatorios contem apenas dados necessarios para a conduta clinica

**Validacao de entradas:**
- Rejeicao de tentativas de prompt injection, SQL injection e command injection
- Limite de comprimento de entradas de texto
- Sanitizacao de caracteres de controle

---

## 8. Limitacoes e Trabalhos Futuros

**Limitacoes atuais:**
- O detector de sangramento e baseado em cor (HSV), com falsos positivos para instrumentos cirurgicos vermelhos. A versao de producao requer YOLOv8 treinado em dataset cirurgico anotado
- Os classificadores de audio nao foram treinados em dados clinicos reais; sao baseados em regras derivadas da literatura. Modelos supervisionados (ex: wav2vec2 fine-tunado) teriam melhor acuracia
- A transcricao com Whisper tem acuracia reduzida em audio de baixa qualidade ou com ruido de fundo medico (equipamentos, alarmes)