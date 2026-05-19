# TFG-NER: Filtrado semiautomático de Entidades Nombradas

Trabajo de Fin de Grado: comparación de tres librerías NER en español 
(spaCy, HuggingFace Transformers y Stanza) sobre dos benchmarks 
neutrales (MultiCoNER v2 y MultiNERD), y desarrollo de una aplicación 
web para el filtrado de candidatos a entidad mediante etiquetado 
morfosintáctico.

**Autor:** Sergio Hernández Rodríguez  
**Tutor:** Mariano Rico  
**Centro:** ETSIINF — Universidad Politécnica de Madrid

## Estructura del repositorio

- **`src/`** — código fuente
  - `evaluar_ner_multiconer.py` — experimento 1, evaluación sobre MultiCoNER v2
  - `evaluar_ner_multinerd.py` — experimento 2, evaluación sobre MultiNERD
  - `tfg-ner-app/` — aplicación web Shiny para el filtrado de candidatos
- **`data/`** — datos de entrada
- **`results/`** — salidas de los experimentos
  - `traza_multiconer.txt` y `resultados_multiconer.txt` — primer experimento
  - `traza_multinerd.txt` y `resultados_multinerd.txt` — segundo experimento
- **`docker/`** — `Dockerfile` para reproducir el entorno
- **`requirements.txt`** — dependencias de Python


## Aplicación de filtrado

La aplicación web permite filtrar listas de candidatos a entidad 
mediante una combinación de tres técnicas: lista negra editable, 
reglas formales (números, URLs, emails, puntuación) y etiquetado 
POS con spaCy.

```bash
cd src/tfg-ner-app
shiny run --reload app.py
```

Abrir el navegador en `http://localhost:8000`.

## Resultados resumidos

| Modelo | MultiCoNER v2 (Strict F1) | MultiNERD (Strict F1) |
|---|---|---|
| spaCy (`es_core_news_lg`) | **0.683** | 0.887 |
| HuggingFace (`mrm8488/bert-spanish-cased-finetuned-ner`) | 0.280 | **0.957** |
| Stanza (`es`) | 0.077 | 0.914 |

El ranking se invierte casi por completo entre los dos datasets, 
lo que indica que el factor responsable es el formato del texto 
(presencia de mayúsculas y ausencia de ruido tipográfico) y no la 
arquitectura del modelo.

Sobre la herramienta de filtrado, sobre un fichero de 1.260 términos 
jurídicos se conservan 884 candidatos (70%) y se descartan 376 (30%), 
con trazabilidad completa del criterio responsable de cada descarte.

## Licencia

Ver fichero `LICENSE`.