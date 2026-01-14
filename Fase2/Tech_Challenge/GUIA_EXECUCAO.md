# Guia Rápido de Execução

## Para o Avaliador/Professor

Este guia mostra as 3 formas mais rápidas de executar e avaliar o projeto.

---

## Opção 1: Executar Script Principal (RECOMENDADO)

**Tempo estimado**: 2-3 minutos

### Passo a Passo:

1. **Navegar até a pasta**:
```powershell
cd "d:\Pos\postech-iaparadevs\Fase2\Tech_Challenge"
```

2. **Criar ambiente virtual** (se ainda não existe):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

3. **Instalar dependências**:
```powershell
pip install -r requirements.txt
```

4. **Executar o sistema**:
```powershell
python src\main.py
```

### O que acontece:
- Carrega pontos de entrega e veículos
- Executa algoritmo genético (300 gerações)
- Gera visualizações (mapas e gráficos)
- Cria relatórios com IA
- Salva tudo em `results/`

### Arquivos gerados:
```
results/
├── graficos/
│   ├── mapa_rotas_otimizadas.html      ← ABRIR NO NAVEGADOR
│   ├── evolucao_algoritmo_genetico.png
│   └── dashboard_interativo.html        ← ABRIR NO NAVEGADOR
└── relatorios/
    ├── instrucoes_motorista_rota_*.txt
    ├── relatorio_executivo.txt
    └── sugestoes_melhoria.txt
```

---

## Opção 2: Jupyter Notebooks (EDUCACIONAL)

**Tempo estimado**: 10-15 minutos (interativo)

### Passo a Passo:

1. **Ativar ambiente e instalar** (se ainda não fez):
```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. **Iniciar Jupyter**:
```powershell
jupyter notebook
```

3. **Abrir notebooks na ordem**:
   - `notebooks/01_introducao_algoritmo_genetico.ipynb` - Conceitos básicos
   - *(Outros notebooks se disponíveis)*

### O que você verá:
- Explicações didáticas sobre Algoritmos Genéticos
- Visualizações interativas
- Código executável célula por célula
- Resultados e análises

---

## Opção 3: Docker (PORTÁVEL)

**Tempo estimado**: 5 minutos

### Passo a Passo:

1. **Construir imagem**:
```powershell
docker build -t sistema-rotas-medicas .
```

2. **Executar container**:
```powershell
docker run -p 8888:8888 sistema-rotas-medicas
```

3. **Acessar**: `http://localhost:8888`

---

## 🧪 Executar Testes

**Verificar qualidade do código**:

```powershell
# Ativar ambiente
.\venv\Scripts\Activate.ps1

# Executar testes
pytest tests/ -v

# Com cobertura
pytest tests/ --cov=src --cov-report=html
```

**Resultado esperado**: Todos os testes passam

---

## Estrutura de Arquivos Importantes

```
Tech_Challenge/
├── src/
│   ├── main.py                    ← EXECUTAR ESTE
│   ├── genetic_algorithm.py       ← Core do AG
│   ├── routing.py                 ← Lógica de rotas
│   ├── visualization.py           ← Mapas e gráficos
│   ├── llm_integration.py         ← Google Gemini
│   └── __init__.py
│
├── notebooks/
│   └── 02_demonstracao_completa.ipynb  ← JUPYTER COMPLETO
│
├── tests/
│   └── test_genetic_algorithm.py  ← TESTES UNITÁRIOS
│
├── data/
│   ├── locais_entrega.csv         ← 31 locais em SP
│   └── medicamentos.csv           ← 30 medicamentos
│
├── results/
│   ├── graficos/                  ← Gerado na execução
│   └── relatorios/                ← Gerado na execução
│
├── README.md                      ← DOCUMENTAÇÃO GERAL
├── RELATORIO_TECNICO.md           ← DETALHES TÉCNICOS
├── GUIA_EXECUCAO.md               ← ESTE ARQUIVO
├── requirements.txt               ← DEPENDÊNCIAS
├── .env.example                   ← Template de configuração
├── Dockerfile                     ← Container Docker
└── .gitignore
```

---

## Configuração da API Google Gemini (OPCIONAL)

Para usar a integração com LLM:

1. **Criar arquivo `.env`** (copiar de `.env.example`):
```powershell
copy .env.example .env
```

2. **Editar `.env`** e adicionar sua chave:
```
GEMINI_API_KEY=sua_chave_aqui
```

**Obtenha sua chave gratuita em:** https://makersuite.google.com/app/apikey

**Nota**: O sistema testa automaticamente 8 modelos Gemini (começando com gemini-2.5-flash-lite) e usa o primeiro disponível

---

## O que Avaliar

### Critérios do Tech Challenge:

1. **Algoritmo Genético**:
   - Implementação completa em `src/genetic_algorithm.py`
   - Operadores: seleção, crossover, mutação
   - Ver evolução em `results/graficos/evolucao_*.png`

2. **Restrições Realistas**:
   - Prioridades (CRITICAL, HIGH, MEDIUM, LOW)
   - Capacidade dos veículos
   - Autonomia/distância máxima
   - Código em `src/routing.py`

3. **Visualizações**:
   - Mapa interativo: `results/graficos/mapa_*.html`
   - Gráficos: `results/graficos/*.png`
   - Dashboard: `results/graficos/dashboard_*.html`

4. **Integração LLM**:
   - Instruções: `results/relatorios/instrucoes_*.txt`
   - Relatórios: `results/relatorios/relatorio_*.txt`
   - Código em `src/llm_integration.py`

5. **Código e Testes**:
   - Arquitetura modular
   - Documentação inline
   - Testes em `tests/`

6. **Documentação**:
   - README.md
   - RELATORIO_TECNICO.md
   - Comentários no código

---

## Solução de Problemas

### Erro: "ModuleNotFoundError"
**Solução**: Certifique-se de que o ambiente virtual está ativado e as dependências instaladas:
```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Erro: "Permission denied" ao ativar venv
**Solução**: Executar como administrador ou ajustar política:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### LLM não funciona
**Solução**: Configure a chave GEMINI_API_KEY no arquivo .env. A API é gratuita (https://makersuite.google.com/app/apikey).

### Jupyter não abre
**Solução**:
```powershell
pip install jupyter notebook
jupyter notebook
```

---

## Informações Adicionais

- **Tempo de execução típico**: 2-3 minutos (300 gerações)
- **Memória requerida**: ~500MB
- **Python**: 3.9+
- **Plataforma**: Windows/Linux/Mac

---

## Dica Final

**Para uma demonstração rápida e completa**:

1. Execute `python src\main.py`
2. Aguarde a conclusão (~3 min)
3. Abra `results/graficos/mapa_rotas_otimizadas.html`
4. Abra `results/graficos/dashboard_interativo.html`
5. Leia `results/relatorios/relatorio_executivo.txt`

Pronto! Você terá uma visão completa do sistema funcionando. 
