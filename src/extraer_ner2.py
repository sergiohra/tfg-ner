
import re
import argparse
from collections import Counter

import spacy


# ── Configuración por defecto ──────────────────────────────────────────────────

DEFAULT_INPUT    = "listadeterminosjur.txt"
DEFAULT_OUTPUT   = "listadeterminosjur_filtrada.txt"
DEFAULT_REJECTED = "listadeterminosjur_rechazados.txt"
DEFAULT_MODEL    = "es_core_news_lg"

# POS tags que indican que un token NO es parte de un nombre propio
NON_NE_POS = {"VERB", "ADV", "ADP", "DET", "CCONJ", "SCONJ", "PRON", "PART", "INTJ", "PUNCT"}


# ── Reglas estructurales ───────────────────────────────────────────────────────

def es_numerico_puro(termino):
    """Dígitos, barras, puntos, comas, guiones — sin ninguna letra."""
    return bool(re.fullmatch(r'[\d\s/.,\-\+\%€$]+', termino))


def es_url_o_email(termino):
    if re.search(r'https?://', termino, re.IGNORECASE):
        return True
    if re.search(r'\bwww\.', termino, re.IGNORECASE):
        return True
    if re.search(r'\S+@\S+\.\S+', termino):
        return True
    return False


def es_fragmento_partido(termino):
    t = termino.strip()
    return t.startswith('"') or t.endswith('"') or t.startswith('\u201c') or t.endswith('\u201d')


# ── Criterio POS con spaCy ─────────────────────────────────────────────────────

def analizar_con_spacy(termino, nlp):
    """
    Analiza el término con spaCy y decide si es NE o no.
    Devuelve (eliminar: bool, razón: str).
    """
    doc = nlp(termino)
    tokens = [t for t in doc if not t.is_space and t.pos_ != "PUNCT"]

    if not tokens:
        return False, ""

    pos_tags = [t.pos_ for t in tokens]
    pos_set  = set(pos_tags)

    # Conservar si hay algún nombre propio o token desconocido/sigla (X)
    if "PROPN" in pos_set or "X" in pos_set:
        return False, ""

    # Término de un solo token
    if len(tokens) == 1:
        pos = pos_tags[0]
        # Solo PROPN y X se conservan en tokens únicos
        if pos not in {"PROPN", "X"}:
            return True, f"token único POS={pos}"
        return False, ""

    # Término multitoken: analizar composición
    non_nominal = [p for p in pos_tags if p in NON_NE_POS]
    nominal     = [p for p in pos_tags if p in {"NOUN", "ADJ", "NUM", "PROPN"}]

    # Si más de la mitad de los tokens son no-nominales → eliminar
    if len(non_nominal) > len(nominal):
        pos_str = ", ".join(pos_tags)
        return True, f"multitoken no-nominal [{pos_str}]"

    # Frase nominal pura → conservar (puede ser entidad de dominio)
    return False, ""


# ── Parseo del fichero ─────────────────────────────────────────────────────────

def parsear_linea(linea):
    match = re.match(r'^(\s*)(.*?):\s*(\d+)\s*$', linea.rstrip("\n"))
    if not match:
        return None
    return match.group(1), match.group(2).strip(), int(match.group(3))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Filtra no-NEs de un fichero de términos con frecuencias usando POS tagging."
    )
    parser.add_argument("--input",    default=DEFAULT_INPUT)
    parser.add_argument("--output",   default=DEFAULT_OUTPUT)
    parser.add_argument("--rejected", default=DEFAULT_REJECTED)
    parser.add_argument("--model",    default=DEFAULT_MODEL,
                        help="Modelo spaCy a usar (debe ser de español)")
    args = parser.parse_args()

    print(f"Cargando modelo spaCy '{args.model}'...")
    # Desactivamos componentes no necesarios para mayor velocidad
    nlp = spacy.load(args.model, disable=["ner", "parser"])
    print("Modelo cargado.\n")

    kept     = []
    rejected = []

    with open(args.input, encoding="utf-8") as f:
        lineas = f.readlines()

    for linea in lineas:
        if not linea.strip():
            continue

        parsed = parsear_linea(linea)
        if parsed is None:
            kept.append((linea.rstrip("\n"), None, ""))
            continue

        indent, termino, freq = parsed

        # Reglas estructurales rápidas (sin modelo)
        if es_url_o_email(termino):
            rejected.append((termino, freq, "URL/email"))
            continue

        if es_numerico_puro(termino):
            rejected.append((termino, freq, "numérico puro"))
            continue

        if es_fragmento_partido(termino):
            rejected.append((termino, freq, "fragmento partido"))
            continue

        # Criterio lingüístico con spaCy
        eliminar, razon = analizar_con_spacy(termino, nlp)
        if eliminar:
            rejected.append((termino, freq, f"POS: {razon}"))
        else:
            kept.append((f"{indent}{termino}: {freq}", freq, ""))

    # Escribir fichero filtrado
    with open(args.output, "w", encoding="utf-8") as f:
        for linea, _, _ in kept:
            f.write(linea + "\n")

    # Escribir fichero de rechazados
    with open(args.rejected, "w", encoding="utf-8") as f:
        f.write(f"{'TÉRMINO':<55} {'FREQ':>6}  RAZÓN\n")
        f.write("-" * 90 + "\n")
        for termino, freq, razon in sorted(rejected, key=lambda x: -x[1]):
            f.write(f"{termino:<55} {freq:>6}  {razon}\n")

    # Resumen
    total = len(kept) + len(rejected)
    print(f"{'='*60}")
    print(f"  Términos totales procesados : {total}")
    print(f"  Términos conservados        : {len(kept)}")
    print(f"  Términos eliminados         : {len(rejected)}")
    print(f"{'='*60}")
    print(f"\n  Fichero filtrado   → {args.output}")
    print(f"  Fichero rechazados → {args.rejected}\n")

    conteo = Counter(r for _, _, r in rejected)
    print("  Desglose de eliminados por categoría:")
    for cat, n in conteo.most_common():
        print(f"    {cat:<55} {n:>4}")
    print()


if __name__ == "__main__":
    main()