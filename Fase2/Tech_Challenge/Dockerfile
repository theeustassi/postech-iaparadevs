FROM python:3.9-slim

WORKDIR /app

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copiar arquivos de requisitos
COPY requirements.txt .

# Instalar dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código fonte
COPY src/ ./src/
COPY notebooks/ ./notebooks/
COPY data/ ./data/

# Criar diretórios para resultados
RUN mkdir -p results/graficos results/relatorios results/rotas_otimizadas

# Expor porta para Jupyter
EXPOSE 8888

# Comando padrão: executar Jupyter
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]
