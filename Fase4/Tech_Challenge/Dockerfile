FROM python:3.11-slim

LABEL maintainer="Matheus Tassi Souza <RM367424>"
LABEL description="Sistema de Monitoramento Multimodal para Saude da Mulher - Tech Challenge Fase 4"

WORKDIR /app

# Dependencias do sistema necessarias para opencv e librosa
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copia e instala dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copia o restante do projeto
COPY . .

# Cria os diretorios de saida
RUN mkdir -p results/graficos results/relatorios results/logs results/modelos \
    data/amostras_video data/amostras_audio

# Por padrao, executa o pipeline completo com dados sinteticos
CMD ["python", "src/main.py"]
