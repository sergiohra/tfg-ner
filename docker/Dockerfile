FROM python:3.10-slim

WORKDIR /app

# Dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Descargar modelos
RUN python -m spacy download es_core_news_lg
RUN python -c "import stanza; stanza.download('es')"

# Copiar código
COPY src/ ./src/

CMD ["python", "src/evaluar_ner.py"]