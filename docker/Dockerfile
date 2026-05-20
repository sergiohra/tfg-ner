FROM python:3.11-slim
WORKDIR /app

# Dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Actualizar pip
RUN pip install --no-cache-dir --upgrade pip

# Dependencias Python con timeout y reintentos
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=300 --retries 5 -r requirements.txt

# Descargar modelos pre-entrenados con versiones fijadas
RUN pip install --no-cache-dir https://github.com/explosion/spacy-models/releases/download/es_core_news_lg-3.8.0/es_core_news_lg-3.8.0-py3-none-any.whl
RUN python -c "import stanza; stanza.download('es')"

# Copiar codigo y datos del proyecto
COPY src/ ./src/
COPY data/ ./data/

# Crear directorio de resultados
RUN mkdir -p /app/results

# Sin CMD por defecto: el usuario elige que script ejecutar