# evaluar_ner.py
# Compara tres librerías NER en español sobre WikiANN
# Librerías: HuggingFace (BERT), spaCy (lg), Stanza
# Métricas: Precision, Recall, F1 (seqeval, evaluación exacta)

import warnings
warnings.filterwarnings("ignore")

import spacy
import stanza
from datasets import load_dataset
from transformers import pipeline
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

# ─────────────────────────────────────────────────────────────
# 1. CARGAR DATASET
# ─────────────────────────────────────────────────────────────
print("Cargando WikiANN (es)...")
dataset = load_dataset("wikiann", "es")

label_names = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
N = len(dataset["test"])  # Número total de frases en el test set (aprox. 10k)

# ─────────────────────────────────────────────────────────────
# 2. INSPECCIÓN DEL DATASET
# ─────────────────────────────────────────────────────────────
print("\n===== INFORMACIÓN DEL DATASET: WikiANN (es) =====")

for split in ["train", "validation", "test"]:
    split_data = dataset[split]
    num_frases = len(split_data)
    total_tokens = sum(len(s) for s in split_data["tokens"])
    total_entidades = sum(
        sum(1 for tag in s if label_names[tag].startswith("B-"))
        for s in split_data["ner_tags"]
    )
    longitud_media = total_tokens / num_frases
    print(f"\n  [{split.upper()}]")
    print(f"    Frases:            {num_frases}")
    print(f"    Tokens totales:    {total_tokens}")
    print(f"    Longitud media:    {longitud_media:.1f} tokens/frase")
    print(f"    Entidades totales: {total_entidades}")

print(f"\n  [TEST — desglose por tipo de entidad]")
conteo = {"PER": 0, "LOC": 0, "ORG": 0}
for ejemplo in dataset["test"]:
    for tag in ejemplo["ner_tags"]:
        etiqueta = label_names[tag]
        if etiqueta.startswith("B-"):
            tipo = etiqueta[2:]
            if tipo in conteo:
                conteo[tipo] += 1
for tipo, n in conteo.items():
    print(f"    {tipo}: {n} entidades")

print(f"\n  Usando para evaluación: {N} frases del test set")
print("=" * 50 + "\n")

# ─────────────────────────────────────────────────────────────
# 3. PREPARAR DATOS DE EVALUACIÓN
# ─────────────────────────────────────────────────────────────
test_data = dataset["test"]
sentences_tokens = [test_data[i]["tokens"] for i in range(N)]
y_true = [
    [label_names[tag] for tag in test_data[i]["ner_tags"]]
    for i in range(N)
]

# ─────────────────────────────────────────────────────────────
# 4. HELPER: spans {start, end, label} → etiquetas IOB por token
# ─────────────────────────────────────────────────────────────
def spans_to_iob(tokens, entities):
    labels = ["O"] * len(tokens)
    offsets, pos = [], 0
    for tok in tokens:
        offsets.append((pos, pos + len(tok)))
        pos += len(tok) + 1

    for ent in entities:
        first = True
        for i, (s, e) in enumerate(offsets):
            if s >= ent["start"] and e <= ent["end"]:
                labels[i] = ("B-" if first else "I-") + ent["label"]
                first = False
    return labels

# ─────────────────────────────────────────────────────────────
# 5. HUGGING FACE — mrm8488/bert-spanish-cased-finetuned-ner
# ─────────────────────────────────────────────────────────────
print("▶ [1/3] HuggingFace (mrm8488/bert-spanish-cased-finetuned-ner)...")
ner_hf = pipeline(
    "ner",
    model="mrm8488/bert-spanish-cased-finetuned-ner",
    aggregation_strategy="simple",
    device=-1  # Cambiar a 0 si tienes GPU
)

y_hf = []
for tokens in sentences_tokens:
    text = " ".join(tokens)
    results = ner_hf(text)
    entities = [
        {"start": r["start"], "end": r["end"], "label": r["entity_group"]}
        for r in results
        if r["entity_group"] in ("PER", "LOC", "ORG")
    ]
    y_hf.append(spans_to_iob(tokens, entities))

# ─────────────────────────────────────────────────────────────
# 6. SPACY — es_core_news_lg
# ─────────────────────────────────────────────────────────────
print("▶ [2/3] spaCy (es_core_news_lg)...")
nlp_spacy = spacy.load("es_core_news_lg")
spacy_map = {"PER": "PER", "LOC": "LOC", "ORG": "ORG", "GPE": "LOC"}

y_spacy = []
for tokens in sentences_tokens:
    text = " ".join(tokens)
    doc = nlp_spacy(text)
    entities = [
        {"start": ent.start_char, "end": ent.end_char, "label": spacy_map[ent.label_]}
        for ent in doc.ents
        if ent.label_ in spacy_map
    ]
    y_spacy.append(spans_to_iob(tokens, entities))

# ─────────────────────────────────────────────────────────────
# 7. STANZA — pipeline español con NER
# ─────────────────────────────────────────────────────────────
print("▶ [3/3] Stanza (es, tokenize+ner)...")
nlp_stanza = stanza.Pipeline('es', processors='tokenize,ner', tokenize_pretokenized=True)

y_stanza = []
for tokens in sentences_tokens:
    doc = nlp_stanza([tokens])
    text = " ".join(tokens)
    entities = []
    for ent in doc.entities:
        if ent.type in ("PER", "LOC", "ORG"):
            start = text.find(ent.text)
            if start != -1:
                entities.append({
                    "start": start,
                    "end": start + len(ent.text),
                    "label": ent.type
                })
    y_stanza.append(spans_to_iob(tokens, entities))

# ─────────────────────────────────────────────────────────────
# 8. TABLA COMPARATIVA
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print(f"  RESULTADOS — WikiANN ES (test, {N} frases, exact match)")
print("=" * 65)
print(f"{'Librería':<35} {'Precision':>9} {'Recall':>9} {'F1':>9}")
print("-" * 65)

modelos = [
    ("HuggingFace (bert-spanish-ner)", y_hf),
    ("spaCy (es_core_news_lg)",        y_spacy),
    ("Stanza (es)",                    y_stanza),
]

for nombre, y_pred in modelos:
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    f = f1_score(y_true, y_pred)
    print(f"{nombre:<35} {p:>9.3f} {r:>9.3f} {f:>9.3f}")

print("=" * 65)

print("\n── Detalle por entidad — HuggingFace ──")
print(classification_report(y_true, y_hf))

print("\n── Detalle por entidad — spaCy ──")
print(classification_report(y_true, y_spacy))

print("\n── Detalle por entidad — Stanza ──")
print(classification_report(y_true, y_stanza))