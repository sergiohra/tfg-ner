## Instalación

### Opción A — Desde DockerHub (recomendada)

```bash
docker pull sergiotfg/tfg-ner:v11
```

### Opción B — Construir la imagen desde el código fuente

```bash
git clone https://github.com/sergiotfg/tfg-ner.git
cd tfg-ner
docker build -t sergiotfg/tfg-ner:v11 -f docker/Dockerfile .
```

La primera construcción puede tardar entre 15 y 30 minutos: instala las dependencias y descarga los modelos de spaCy y Stanza.

### Opción C — Instalación local (sin Docker)

```bash
git clone https://github.com/sergiotfg/tfg-ner.git
cd tfg-ner
pip install -r requirements.txt
python -m spacy download es_core_news_lg
python -c "import stanza; stanza.download('es')"
```

---

## Uso

Como la imagen no define un `CMD` por defecto, cada comando indica explícitamente qué ejecutar. La opción `-v` monta la carpeta `results/` local sobre la del contenedor para conservar las salidas.

> **Windows (PowerShell):** sustituir `$(pwd)` por `${PWD}` y escribir cada comando en una sola línea.

### Experimento 1 — Evaluación en MultiCoNER v2

Banco de pruebas principal: subconjunto en español de MultiCoNER v2 (SemEval-2023). Benchmark neutral, ninguna librería fue entrenada sobre él.

```bash
docker run --rm -v "$(pwd)/results:/app/results" \
    sergiotfg/tfg-ner:v11 python src/evaluar_ner_multiconer.py
```

Genera en `results/` los ficheros `resultados_multiconer.txt` (métricas) y `traza_multiconer.txt` (log de ejemplos). Tiempo aproximado: 20-25 minutos en CPU (Stanza es la fase más lenta, ~17 min).

### Experimento 2 — Evaluación en MultiNERD

Mismas tres librerías sobre MultiNERD (Tedeschi & Navigli, 2022), con capitalización normal, para contrastar el efecto de las minúsculas observado en MultiCoNER v2.

```bash
docker run --rm -v "$(pwd)/results:/app/results" \
    sergiotfg/tfg-ner:v11 python src/evaluar_ner_multinerd.py
```

Genera `resultados_multinerd.txt` y `traza_multinerd.txt` en `results/`.

### Aplicación Shiny — Filtrado de candidatos a entidad

Aplicación web que filtra un fichero de términos candidatos a entidad y elimina falsos positivos. Combina tres técnicas activables de forma independiente: una **lista negra editable**, un conjunto de **reglas formales** (número, URL, email, puntuación descuadrada y palabras trampa) y un criterio de **etiquetado morfosintáctico (POS)** con spaCy. La salida muestra los términos conservados y los eliminados, junto con el motivo de cada descarte. Pensada para usuarios sin formación técnica.

```bash
docker run --rm -p 8000:8000 -w /app/src/tfg-ner-app \
    sergiotfg/tfg-ner:v11 \
    shiny run --host 0.0.0.0 --port 8000 app.py
```

Una vez lanzada, abrir en el navegador del host:
http://localhost:8000

El flag `-p 8000:8000` expone el puerto al host y `-w` sitúa el directorio de trabajo en la carpeta de la app (necesario para que cargue correctamente sus recursos estáticos).

---

## Resultados principales

Resultados sobre el conjunto de **test** de MultiCoNER v2 (español), modo `strict`:

| Librería | F1 |
|---|---:|
| **spaCy** (`es_core_news_lg`) | **0.683** |
| **HuggingFace** (`mrm8488/bert-spanish-cased-finetuned-ner`) | 0.280 |
| **Stanza** (`es`) | 0.077 |

Sobre MultiNERD (capitalización normal) el ranking se invierte casi por completo (BERT 0.957, Stanza 0.914, spaCy 0.887), lo que evidencia que el factor dominante de degradación es el formato del texto de entrada, no la arquitectura. Los valores completos de Precision/Recall por modo y por tipo de entidad se generan al ejecutar los scripts y se guardan en `results/`.

---

## Reproducibilidad

- La imagen Docker fija las versiones de Python (3.11), spaCy (3.8.0) y el resto de dependencias declaradas en `requirements.txt`.
- El modelo `es_core_news_lg` se instala desde una URL versionada, evitando actualizaciones silenciosas.
- Los corpus de evaluación (MultiCoNER v2 y MultiNERD) se descargan desde HuggingFace en el momento de la ejecución; la lista de términos jurídicos de la aplicación está incluida en `data/`.
- Los volúmenes Docker (`-v ...:/app/results`) permiten conservar las salidas fuera del contenedor.

Para replicar los resultados publicados:

```bash
docker pull sergiotfg/tfg-ner:v11
docker run --rm -v "$(pwd)/results:/app/results" sergiotfg/tfg-ner:v11 python src/evaluar_ner_multiconer.py
docker run --rm -v "$(pwd)/results:/app/results" sergiotfg/tfg-ner:v11 python src/evaluar_ner_multinerd.py
```

---

## Licencia

Este proyecto se distribuye bajo licencia MIT. Consulta el archivo [LICENSE](LICENSE) para más detalles.
