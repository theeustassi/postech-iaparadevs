# Sistema de Otimização de Rotas para Distribuição de Medicamentos e Insumos Médicos

## Tech Challenge - Fase 2 | FIAP Pós-Tech IA para Devs

Este é o projeto da Fase 2 do Tech Challenge.

Neste projeto, foi desenvolvido um sistema inteligente para otimizar rotas de entrega de medicamentos e insumos médicos usando **Algoritmos Genéticos** e integração com **LLMs (Large Language Models)** para geração de relatórios e instruções.

O sistema resolve o problema do "caixeiro viajante médico", considerando restrições realistas como:
- Prioridades de entrega (medicamentos críticos vs. insumos regulares)
- Capacidade limitada dos veículos
- Autonomia/distância máxima que cada veículo pode percorrer
- Múltiplos veículos disponíveis
- Janelas de tempo para entregas

## Funcionalidades do Sistema

1. **Otimiza rotas** usando algoritmos genéticos para encontrar a melhor sequência de entregas
2. **Visualiza as rotas** em mapas interativos para fácil interpretação
3. **Gera instruções detalhadas** para motoristas usando LLMs
4. **Cria relatórios inteligentes** sobre eficiência, economia de tempo e recursos
5. **Responde perguntas** em linguagem natural sobre as rotas e entregas

## Estrutura do Projeto

```
Tech_Challenge/
├── notebooks/
│   └── 02_demonstracao_completa.ipynb
├── src/
│   ├── __init__.py
│   ├── genetic_algorithm.py      # Implementacao do AG
│   ├── routing.py                # Logica de roteamento
│   ├── visualization.py          # Visualizacao de rotas
│   ├── llm_integration.py        # Integracao com Google Gemini
│   └── main.py                   # Script principal
├── data/
│   ├── locais_entrega.csv        # 31 locais em Sao Paulo
│   └── medicamentos.csv          # 30 medicamentos e insumos
├── results/
│   ├── graficos/                 # Mapas e graficos (gerado na execucao)
│   └── relatorios/               # Relatorios LLM (gerado na execucao)
├── tests/
│   └── test_genetic_algorithm.py # Testes unitarios
├── Dockerfile                    # Container Docker
├── requirements.txt              # Dependencias Python
├── .env.example                  # Template para configuracao
├── .gitignore                    # Arquivos ignorados pelo Git
├── RELATORIO_TECNICO.md          # Relatorio tecnico completo
├── GUIA_EXECUCAO.md              # Guia rapido de execucao
└── README.md                     # Documentacao principal
```

## Como Usar

### Pré-requisitos
- Python 3.9+
- Uma chave de API Google Gemini (gratuita - para integracao com LLM)
- Docker (opcional)

### Instalação

#### Opção 1: Instalação Local

**1. Clone o repositório e navegue até a pasta:**
```bash
cd Fase2/Tech_Challenge
```

**2. Crie e ative um ambiente virtual:**

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
```

**3. Instale as dependências:**
```bash
pip install -r requirements.txt
```

**4. Configure a chave da API Google Gemini:**
- Copie o arquivo `.env.example` para `.env`
- Adicione sua chave de API Google Gemini no arquivo `.env`:
```
GEMINI_API_KEY=sua_chave_aqui
```

**Obtenha sua chave gratuita em:** https://makersuite.google.com/app/apikey

### Executando o Sistema

#### Via Jupyter Notebooks (Recomendado para aprendizado)

```bash
jupyter notebook
```

Abra os notebooks na ordem:
1. `01_introducao_algoritmo_genetico.ipynb` - Entenda os conceitos básicos
2. `02_demonstracao_completa.ipynb` - Veja o sistema completo em ação

#### Via Script Python

```bash
python src/main.py
```

#### Via Docker

```bash
# Construir a imagem
docker build -t sistema-rotas-medicas .

# Executar o container
docker run -p 8888:8888 sistema-rotas-medicas
```

## Como Funciona o Algoritmo Genético?

O sistema usa conceitos da evolução natural:

1. **População Inicial**: Gera rotas aleatórias
2. **Avaliação (Fitness)**: Calcula quão boa é cada rota considerando:
   - Distância total percorrida
   - Prioridades das entregas
   - Capacidade dos veículos
   - Autonomia/combustível
3. **Seleção**: Escolhe as melhores rotas
4. **Cruzamento (Crossover)**: Combina partes de rotas boas
5. **Mutação**: Introduz pequenas mudanças aleatórias
6. **Nova Geração**: Repete o processo até encontrar a melhor solução

## Integração com LLM

O sistema utiliza **Google Gemini** (gratuito) para gerar relatórios inteligentes:

- **Modelo**: gemini-2.5-flash-lite com fallback automático
- **Fallback inteligente**: Testa automaticamente múltiplos modelos até encontrar um disponível
- **Relatórios em português** sobre rotas otimizadas  
- **Instruções detalhadas** para motoristas
- **Sugestões de melhoria** baseadas em análise de dados
- **Limite de tokens**: 8192 (respostas completas sem cortes)

### Configurar API Gemini (Gratuita)

1. Obtenha chave gratuita em: https://makersuite.google.com/app/apikey
2. Crie arquivo `.env` na raiz do projeto:
```
GEMINI_API_KEY=sua_chave_aqui
```
3. Execute o sistema - o fallback automático encontrará o melhor modelo disponível

O sistema usa Google Gemini (gemini-2.5-flash-lite) para:
- Gerar instruções passo a passo para motoristas
- Criar relatórios executivos sobre desempenho
- Responder perguntas sobre as rotas em linguagem natural
- Sugerir melhorias no processo

## Resultados

O sistema gera:
- Mapas interativos das rotas otimizadas
- Gráficos de evolução do algoritmo genético
- Comparativo de desempenho antes/depois da otimização
- Relatórios detalhados de economia de tempo e recursos

## Testes

Execute os testes automatizados:
```bash
pytest tests/ -v
```

## Documentação Adicional

- [RELATORIO_TECNICO.md](RELATORIO_TECNICO.md) - Relatório técnico detalhado
- [CODIGO_BASE.md](CODIGO_BASE.md) - Informações sobre código base da FIAP
- [GUIA_EXECUCAO.md](GUIA_EXECUCAO.md) - Guia rápido de execução
- [Notebooks](notebooks/) - Exemplos práticos e tutoriais

## Desenvolvido por

Matheus Tassi Souza - RM367424
Fase 2 - Evolução da IA: GenAI, Cloud ML e LLMs

---

**Nota**: Este é um projeto educacional. Para uso em produção em ambientes hospitalares reais, seria necessário validação adicional, testes de segurança e conformidade com regulamentações da área da saúde.
