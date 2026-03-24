FROM python:3.11-slim

LABEL maintainer="Matheus Tassi Souza <RM367424>"
LABEL description="Assistente Medico Virtual HospitalIQ - Tech Challenge Fase 3"

WORKDIR /app

# Instala dependencias do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copia e instala dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copia o restante do projeto
COPY . .

# Cria os diretorios de saida
RUN mkdir -p results/data results/modelos results/logs results/graficos

# Expoe a porta para o Jupyter (quando necessario)
EXPOSE 8888

# Por padrao, executa o pipeline principal com modo demo
CMD ["python", "src/main.py", "--demo-only"]
