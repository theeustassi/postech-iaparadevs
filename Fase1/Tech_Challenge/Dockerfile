# Dockerfile para Tech Challenge - Diagnóstico de Câncer de Mama

FROM python:3.9-slim

# Define o diretório de trabalho
WORKDIR /app

# Instala dependências do sistema
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copia os arquivos de requisitos
COPY requirements.txt .

# Instala as dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Copia todo o projeto
COPY . .

# Cria diretórios necessários
RUN mkdir -p data results/graficos

# Expõe a porta do Jupyter
EXPOSE 8888

# Comando padrão: iniciar Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
