import warnings
warnings.filterwarnings("ignore")

import time
import os
import spacy
import stanza
from datasets import load_dataset
from transformers import pipeline
from nervaluate import Evaluator
from collections import Counter

# Configuración del experimento
TRAZA_FILE = "results/traza_multiconer.txt"  
RESULTADOS_FILE = "results/resultados_multiconer.txt"
N = 5000 
NUM_EJEMPLOS_TRAZA = 20 
BATCH_SIZE_HF = 32 
BATCH_SIZE_SPACY = 500 

# Modos de evaluación para nervaluate
MODES = ["strict", "exact", "partial", "ent_type"]
MODE_NAMES = {"strict": "strict", "exact": "exact", "partial": "partial", "ent_type": "type"}

def log(msg, fh):
    print(msg)
    fh.write(msg + "\n")

FINE_TO_COARSE = {
    "Scientist": "PER", "Artist": "PER", "Athlete": "PER",
    "Politician": "PER", "Cleric": "PER", "SportsManager": "PER",
    "OtherPER": "PER",
    "Facility": "LOC", "OtherLOC": "LOC",
    "HumanSettlement": "LOC", "Station": "LOC",
    "MusicalGRP": "ORG", "PublicCORP": "ORG", "PrivateCORP": "ORG",
    "AerospaceManufacturer": "ORG", "SportsGRP": "ORG",
    "CarManufacturer": "ORG", "ORG": "ORG",
}

KEEP_TYPES = {"PER", "LOC", "ORG"}

def map_tag(tag):
    if tag == "O":
        return "O"
    prefix, tipo = tag.split("-", 1)
    coarse = FINE_TO_COARSE.get(tipo)
    if coarse:
        return f"{prefix}-{coarse}"
    return "O"


def spans_to_iob(tokens, entities):
    labels = ["O"] * len(tokens)

    offsets = []
    pos = 0
    for tok in tokens:
        offsets.append((pos, pos + len(tok)))
        pos += len(tok) + 1

    for ent in entities:
        first = True
        for i, (s, e) in enumerate(offsets):
            if not (e <= ent["start"] or s >= ent["end"]):
                labels[i] = ("B-" if first else "I-") + ent["label"]
                first = False

    return labels



ft = open(TRAZA_FILE, "w", encoding="utf-8")
log("=" * 70, ft)
log("  TRAZA DE EJECUCIÓN — MultiCoNER v2 (es)", ft)
log("=" * 70, ft)
log("\nCargando MultiCoNER v2 (español)...", ft)
dataset = load_dataset("MultiCoNER/multiconer_v2", "Spanish (ES)")


log("\n===== INFORMACIÓN DEL DATASET =====", ft)
for split in ["train", "validation", "test"]:
    split_data = dataset[split]
    num_frases = len(split_data)
    total_tokens = sum(len(s) for s in split_data["tokens"])
    longitud_media = total_tokens / num_frases if num_frases > 0 else 0
    total_ent = sum(1 for ej in split_data for tag in ej["ner_tags"] if map_tag(tag).startswith("B-"))
    log(f"\n  [{split.upper()}]", ft)
    log(f"    Frases: {num_frases}  |  Tokens: {total_tokens}  |  Media: {longitud_media:.1f} tok/frase  |  Entidades PER/LOC/ORG: {total_ent}", ft)

conteo = {"PER": 0, "LOC": 0, "ORG": 0}
conteo_fine = Counter()
for ej in dataset["test"]:
    for tag in ej["ner_tags"]:
        if tag.startswith("B-"):
            tipo_fine = tag[2:]
            coarse = FINE_TO_COARSE.get(tipo_fine)
            if coarse:
                conteo[coarse] += 1
                conteo_fine[tipo_fine] += 1

log(f"\n  [TEST coarse]  PER: {conteo['PER']}  |  LOC: {conteo['LOC']}  |  ORG: {conteo['ORG']}", ft)
log(f"  [TEST fine-grained top 10]", ft)
for tipo, n in conteo_fine.most_common(10):
    log(f"    {tipo} ({FINE_TO_COARSE.get(tipo,'?')}): {n}", ft)

test_data = dataset["test"]
if N > len(test_data):
    N = len(test_data)
log(f"\n  Evaluando: {N} frases", ft)

# Preparar los datos de entrada y las etiquetas para la evaluación
sentences_tokens = [test_data[i]["tokens"] for i in range(N)]
sentences_text = [" ".join(tokens) for tokens in sentences_tokens]
y_true = [[map_tag(tag) for tag in test_data[i]["ner_tags"]] for i in range(N)]

log("\n--- EJEMPLOS DE MAPEO ---", ft)
for i in range(min(5, N)):
    log(f"  Frase {i}: {sentences_text[i][:80]}{'...' if len(sentences_text[i]) > 80 else ''}", ft)
    log(f"    Original: {test_data[i]['ner_tags'][:12]}", ft)
    log(f"    Mapeado:  {y_true[i][:12]}", ft)


log(f"\n▶ [1/3] bert-spanish-cased-finetuned-ner — batch_size={BATCH_SIZE_HF}...", ft)
t0 = time.time()
ner_hf = pipeline("ner", model="mrm8488/bert-spanish-cased-finetuned-ner",
                  aggregation_strategy="simple", device=-1, batch_size=BATCH_SIZE_HF)

y_hf = []
all_hf_entities = []
hf_results_all = ner_hf(sentences_text)

for idx, (tokens, results) in enumerate(zip(sentences_tokens, hf_results_all)):
    entities_raw = [
        {"start": r["start"], "end": r["end"], "label": r["entity_group"],
         "score": r["score"], "word": r["word"]}
        for r in results if r["entity_group"] in KEEP_TYPES
    ]
    entities = [{"start": e["start"], "end": e["end"], "label": e["label"]} for e in entities_raw]
    iob = spans_to_iob(tokens, entities)
    y_hf.append(iob)
    all_hf_entities.append(entities_raw)

t_hf = time.time() - t0
log(f"   Tiempo: {t_hf:.1f} s ({N/t_hf:.0f} frases/s)", ft)

for idx in range(min(NUM_EJEMPLOS_TRAZA, N)):
    log(f"\n  [BERT] Frase {idx}: {sentences_text[idx][:80]}{'...' if len(sentences_text[idx]) > 80 else ''}", ft)
    for e in all_hf_entities[idx]:
        log(f"    → '{e['word']}' = {e['label']} (score: {e['score']:.3f})", ft)
    log(f"    Predicho: {y_hf[idx]}", ft)
    log(f"    Real:     {y_true[idx]}", ft)


log(f"\n▶ [2/3] spaCy — batch_size={BATCH_SIZE_SPACY}, nlp.pipe()...", ft)
t0 = time.time()
nlp_spacy = spacy.load("es_core_news_lg")
spacy_map = {"PER": "PER", "LOC": "LOC", "ORG": "ORG", "GPE": "LOC"}

y_spacy = []
all_spacy_entities = []

for idx, doc in enumerate(nlp_spacy.pipe(sentences_text, batch_size=BATCH_SIZE_SPACY)):
    tokens = sentences_tokens[idx]
    entities_raw = [
        {"start": ent.start_char, "end": ent.end_char,
         "label": spacy_map[ent.label_], "text": ent.text,
         "label_orig": ent.label_}
        for ent in doc.ents if ent.label_ in spacy_map
    ]
    entities = [{"start": e["start"], "end": e["end"], "label": e["label"]} for e in entities_raw]
    iob = spans_to_iob(tokens, entities)
    y_spacy.append(iob)
    all_spacy_entities.append(entities_raw)

t_spacy = time.time() - t0
log(f"   Tiempo: {t_spacy:.1f} s ({N/t_spacy:.0f} frases/s)", ft)

for idx in range(min(NUM_EJEMPLOS_TRAZA, N)):
    log(f"\n  [spaCy] Frase {idx}: {sentences_text[idx][:80]}{'...' if len(sentences_text[idx]) > 80 else ''}", ft)
    for e in all_spacy_entities[idx]:
        log(f"    → '{e['text']}' = {e['label']} (original: {e['label_orig']})", ft)
    log(f"    Predicho: {y_spacy[idx]}", ft)
    log(f"    Real:     {y_true[idx]}", ft)


log(f"\n▶ [3/3] Stanza (es, tokenize+ner)...", ft)
t0 = time.time()
nlp_stanza = stanza.Pipeline('es', processors='tokenize,ner', tokenize_pretokenized=True)

y_stanza = []
all_stanza_entities = []

for idx, tokens in enumerate(sentences_tokens):
    if len(tokens) == 0:
        y_stanza.append([])
        all_stanza_entities.append([])
        continue
    doc = nlp_stanza([tokens])
    text = " ".join(tokens)
    entities_raw = []
    search_start = 0
    for ent in doc.entities:
        if ent.type in KEEP_TYPES:
            start = text.find(ent.text, search_start)
            if start != -1:
                entities_raw.append({"start": start, "end": start + len(ent.text),
                                     "label": ent.type, "text": ent.text})
                search_start = start + len(ent.text)
    entities = [{"start": e["start"], "end": e["end"], "label": e["label"]} for e in entities_raw]
    iob = spans_to_iob(tokens, entities)
    y_stanza.append(iob)
    all_stanza_entities.append(entities_raw)

    if (idx + 1) % 1000 == 0:
        elapsed = time.time() - t0
        print(f"   Stanza: {idx+1}/{N} ({elapsed:.0f} s)")

t_stanza = time.time() - t0
log(f"   Tiempo: {t_stanza:.1f} s ({N/t_stanza:.0f} frases/s)", ft)

for idx in range(min(NUM_EJEMPLOS_TRAZA, N)):
    log(f"\n  [Stanza] Frase {idx}: {sentences_text[idx][:80]}{'...' if len(sentences_text[idx]) > 80 else ''}", ft)
    for e in all_stanza_entities[idx]:
        log(f"    → '{e['text']}' = {e['label']}", ft)
    log(f"    Predicho: {y_stanza[idx]}", ft)
    log(f"    Real:     {y_true[idx]}", ft)

log(f"\n{'=' * 70}", ft)
log(f"  FIN DE LA TRAZA — {N} frases procesadas", ft)
log(f"{'=' * 70}", ft)
ft.close()
print(f"\n✓ Traza guardada en: {os.path.abspath(TRAZA_FILE)}")


fr = open(RESULTADOS_FILE, "w", encoding="utf-8")

def res(msg):
    print(msg)
    fr.write(msg + "\n")

res("=" * 70)
res(f"  RESULTADOS — MultiCoNER v2 ES ({N} frases, nervaluate)")
res("=" * 70)

modelos = [
    ("bert-spanish-cased-finetuned-ner", y_hf, t_hf),
    ("spaCy (es_core_news_lg)", y_spacy, t_spacy),
    ("Stanza (es)", y_stanza, t_stanza),
]
tags = ["PER", "LOC", "ORG"]

for nombre, y_pred, t_exec in modelos:
    res(f"\n{'─' * 70}")
    res(f"  {nombre}  (tiempo: {t_exec:.1f} s)")
    res(f"{'─' * 70}")

    evaluator = Evaluator(y_true, y_pred, tags=tags, loader="list")
    result = evaluator.evaluate()
    overall = result["overall"]
    entities = result["entities"]

    # Tabla de los 4 modos de evaluación
    res(f"\n  {'Modo':<12} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    res(f"  {'-' * 44}")
    for mode in MODES:
        ev = overall[mode]
        res(f"  {MODE_NAMES[mode]:<12} {ev.precision:>10.3f} {ev.recall:>10.3f} {ev.f1:>10.3f}")

    # Desglose por tipo de entidad (modo strict)
    res(f"\n  Desglose por entidad (strict):")
    res(f"  {'Tipo':<8} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    res(f"  {'-' * 50}")
    for tag in tags:
        if tag in entities:
            ev = entities[tag]["strict"]
            res(f"  {tag:<8} {ev.precision:>10.3f} {ev.recall:>10.3f} {ev.f1:>10.3f} {ev.possible:>10}")

    # Análisis de errores
    res(f"\n  Análisis de errores:")
    res(f"  {'Modo':<12} {'Correct':>8} {'Incorrect':>10} {'Partial':>8} {'Missed':>8} {'Spurious':>9}")
    res(f"  {'-' * 58}")
    for mode in MODES:
        ev = overall[mode]
        res(f"  {MODE_NAMES[mode]:<12} {ev.correct:>8} {ev.incorrect:>10} {ev.partial:>8} {ev.missed:>8} {ev.spurious:>9}")

# Tabla resumen
res(f"\n{'=' * 80}")
res(f"  RESUMEN COMPARATIVO — MultiCoNER v2 ES ({N} frases)")
res(f"{'=' * 80}")
res(f"\n  {'Librería':<28} {'Strict F1':>10} {'Exact F1':>10} {'Partial F1':>10} {'Type F1':>10} {'Tiempo':>10}")
res(f"  {'-' * 80}")
for nombre, y_pred, t_exec in modelos:
    evaluator = Evaluator(y_true, y_pred, tags=tags, loader="list")
    result = evaluator.evaluate()
    overall = result["overall"]
    sf = overall["strict"].f1
    ef = overall["exact"].f1
    pf = overall["partial"].f1
    tf = overall["ent_type"].f1
    res(f"  {nombre:<28} {sf:>10.3f} {ef:>10.3f} {pf:>10.3f} {tf:>10.3f} {t_exec:>8.1f} s")

res(f"\n{'=' * 80}")
res("""
LEYENDA de modos de evaluación (nervaluate):
  strict  = límites Y tipo deben coincidir exactamente
  exact   = límites deben coincidir, tipo puede ser diferente
  partial = crédito parcial si los límites solapan parcialmente
  type    = tipo debe coincidir, límites pueden solapar parcialmente

LEYENDA de errores:
  correct   = entidades correctas (TP)
  incorrect = tipo o límites incorrectos
  partial   = límites parcialmente correctos
  missed    = entidades del gold standard no detectadas (FN)
  spurious  = entidades predichas que no existen en gold standard (FP)
""")

fr.close()
print(f"\n✓ Resultados guardados en: {os.path.abspath(RESULTADOS_FILE)}")
print(f"✓ Traza guardada en: {os.path.abspath(TRAZA_FILE)}")
