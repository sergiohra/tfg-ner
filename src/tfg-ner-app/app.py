import re
from pathlib import Path
from faicons import icon_svg
import pandas as pd
import spacy
from shiny import App, reactive, render, ui


_NLP = None


def get_nlp():
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("es_core_news_lg", disable=["ner", "parser"])
    return _NLP


DEFAULT_BLACKLIST = """lunes a viernes
cl@ve pin
día siguiente
real
código
teléfono de lunes
servicio
número de días
horas extraor
número siguiente
plan
tiempo completo
mes posterior
ción
días a partir
véanse días"""


BLACKLIST_PATH = Path(__file__).parent / "blacklist.txt"


def cargar_blacklist() -> str:
    """
    Lee la lista negra desde disco. Si el fichero no existe (primera
    ejecución), devuelve la lista por defecto.
    """
    if BLACKLIST_PATH.exists():
        return BLACKLIST_PATH.read_text(encoding="utf-8")
    return DEFAULT_BLACKLIST


def guardar_blacklist(texto: str) -> None:
    """
    Guarda la lista negra en disco (sobreescribe el fichero anterior).
    """
    BLACKLIST_PATH.write_text(texto, encoding="utf-8")


def parse_input_file(text: str) -> pd.DataFrame:
    items = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if ":" in line:
            term, _, freq_str = line.rpartition(":")
            term = term.strip()
            try:
                freq = int(freq_str.strip())
            except ValueError:
                term, freq = line, 1
        else:
            term, freq = line, 1
        if term:
            items.append({"término": term, "frecuencia": freq})
    return pd.DataFrame(items)


_RE_NUM = re.compile(r"^[\d.,/\-\s]+$")
_RE_URL = re.compile(r"https?://|www\.", re.IGNORECASE)
_RE_EMAIL = re.compile(r"\S+@\S+\.\S+")



def regla_numero(t: str) -> str | None:
    if _RE_NUM.match(t.strip()):
        return "Reglas: número "
    return None


def regla_url(t: str) -> str | None:
    if _RE_URL.search(t):
        return "Reglas: URL"
    return None


def regla_email(t: str) -> str | None:
    if _RE_EMAIL.search(t):
        return "Reglas: email"
    return None


def regla_puntuacion_descuadrada(t: str) -> str | None:
    # Comillas dobles de todo tipo
    dobles = t.count('"') + t.count('”') + t.count('“')
    # Comillas angulares (deben balancearse entre sí)
    angulares = t.count('«') - t.count('»')

    # Paréntesis
    parentesis_desbalanceados = t.count("(") != t.count(")")
    # Corchetes
    corchetes_desbalanceados = t.count("[") != t.count("]")

    if dobles % 2 != 0 or angulares != 0 or parentesis_desbalanceados or corchetes_desbalanceados:
        return "Reglas: fragmento (puntuación descuadrada)"
    return None


_PALABRAS_TRAMPA = {"través" , "véanse"}


def regla_palabras_trampa(t: str) -> str | None:
    tokens = set(t.lower().split())
    interseccion = tokens & _PALABRAS_TRAMPA
    if interseccion:
        palabra = next(iter(interseccion))
        return f"Reglas: contiene «{palabra}»"
    return None



REGLAS_DISPONIBLES = {
    "numero":     ("Número",                    regla_numero),
    "url":        ("URL",                            regla_url),
    "email":      ("Email",                          regla_email),
    "puntuacion": ("Puntuación descuadrada",         regla_puntuacion_descuadrada),
    "palabras_trampa": ("Palabras trampa",            regla_palabras_trampa),
}

DESCRIPCIONES_REGLAS = {
    "numero":     "Elimina términos formados solo por dígitos, comas, "
                  "puntos, barras o guiones (ej. 3,5 o 2024/01).",
    "url":        "Elimina términos que contienen http://, https:// o www.",
    "email":      "Elimina términos con formato de email "
                  "(texto@dominio.com).",
    "puntuacion": "Elimina términos con comillas o paréntesis "
                  "desbalanceados, indicador de fragmentos de oración.",

    "palabras_trampa": "Elimina términos que contienen palabras que "
                       "suelen indicar fragmentos de oración, como "
                       "«véanse» o «través»."
}

DESCRIPCION_POS = (
    "Analiza la categoría gramatical de cada término con spaCy. "
    "Conserva nombres propios, siglas y construcciones nominales. "
    "Elimina palabras comunes aisladas y fragmentos con verbos."
)

DESCRIPCION_BLACKLIST = (
    "Aplica la lista negra editable. Los términos que coincidan con "
    "alguna línea de la lista se eliminan independientemente del resto "
    "de filtros."
)

DESCRIPCION_ARCHIVO = (
    "Fichero de texto plano (.txt) con un término por línea. "
    "Cada línea puede ser un término simple o seguir el formato «término: frecuencia»\n"
    "Ejemplos: «aeat: 87», «contrato de trabajo: 19» o «madrid»."
)

_POS_NOMINAL = {"NOUN", "PROPN", "ADJ", "DET", "ADP", "X", "NUM"}
_POS_CONSERVAR_SOLO = {"PROPN", "X"}
_PALABRAS_TRAMPA = {"través" , "véanse"}

def checkbox_con_info(id_, etiqueta, descripcion):

    return ui.div(
        ui.input_checkbox(id_, etiqueta, value=True),
        ui.tooltip(
            ui.span(
                icon_svg("circle-info"),
                style="margin-left: 0.5rem; color: #6c757d; "
                      "cursor: help;",
            ),
            descripcion,
        ),
        style="display: flex; align-items: center;",
    )

def aplicar_pos_tagging(term: str, nlp_model) -> str | None:
    doc = nlp_model(term)
    pos_tags = [tok.pos_ for tok in doc if not tok.is_punct and not tok.is_space]
    if not pos_tags:
        return "POS: sin tokens analizables"
    if len(pos_tags) == 1:
        if pos_tags[0] not in _POS_CONSERVAR_SOLO:
            return f"POS: token único ({pos_tags[0]})"
        return None
    if any(p in _POS_CONSERVAR_SOLO for p in pos_tags):
        return None
    if all(p in _POS_NOMINAL for p in pos_tags):
        return None
    return f"POS: estructura no nominal ({'/'.join(pos_tags)})"


def aplicar_blacklist(term: str, blacklist_set: set) -> str | None:
    if term.lower().strip() in blacklist_set:
        return "Lista negra"
    return None


def filtrar(df, reglas_activas, usar_pos, usar_blacklist, blacklist_set):

    if df.empty:
        return df.copy(), df.copy().assign(motivo=pd.Series(dtype=str))

    nlp_model = get_nlp() if usar_pos else None

    motivos = []
    for term in df["término"]:
        motivo = None

        if usar_blacklist:
            motivo = aplicar_blacklist(term, blacklist_set)

        if motivo is None:
            for rid, (_label, fn) in REGLAS_DISPONIBLES.items():
                if rid in reglas_activas:
                    motivo = fn(term)
                    if motivo is not None:
                        break

        if motivo is None and usar_pos and nlp_model is not None:
            motivo = aplicar_pos_tagging(term, nlp_model)

        motivos.append(motivo)

    df = df.copy()
    df["motivo"] = motivos
    df["Número de tokens"] = df["término"].str.split().str.len()
    conservados = df[df["motivo"].isna()].drop(columns=["motivo"]).reset_index(drop=True)
    eliminados = df[df["motivo"].notna()].reset_index(drop=True)
    return conservados, eliminados




app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.h4("Configuración del filtrado"),
        ui.div(
    ui.input_file(
        "archivo",
        "Archivo de términos",
        accept=[".txt"],
        multiple=False,
        placeholder="ningún archivo seleccionado",
    ),
    ui.tooltip(
        ui.span(
            icon_svg("circle-info"),
            style="margin-left: 0.5rem; color: #6c757d; "
                  "cursor: help; align-self: flex-end; "
                  "margin-bottom: 0.7rem;",
        ),
        DESCRIPCION_ARCHIVO,
    ),
    style="display: flex; align-items: center;",
),
        ui.tags.hr(),
       ui.h6("Reglas"),
*[
    checkbox_con_info(rid, label, DESCRIPCIONES_REGLAS.get(rid, ""))
    for rid, (label, _fn) in REGLAS_DISPONIBLES.items()
],
        ui.tags.hr(),
       ui.div(
        ui.input_switch("usar_pos", "Filtrado con spaCy", value=True),
        ui.tooltip(
            ui.span(
                icon_svg("circle-info"),
                style="margin-left: 0.5rem; color: #6c757d; cursor: help;",
            ),
            DESCRIPCION_POS,
        ),
        style="display: flex; align-items: center;",
    ),
        ui.tags.hr(),
       ui.div(
        ui.input_switch("usar_blacklist", "Aplicar lista negra", value=True),
        ui.tooltip(
            ui.span(
                icon_svg("circle-info"),
                style="margin-left: 0.5rem; color: #6c757d; cursor: help;",
            ),
            DESCRIPCION_BLACKLIST,
        ),
        style="display: flex; align-items: center;",
    ),
        ui.input_text_area(
            "blacklist",
            "Términos de la lista negra",
            value=cargar_blacklist(),
            rows=10,
            width="100%",
            update_on="blur",
        ),
        ui.help_text(
            "Edita la lista y haz clic fuera del recuadro para aplicar."
        ),
        width=380,
    ),
    ui.h2("Filtrador interactivo de términos NER"),
    ui.p(
        "Filtrado de términos para limpieza de datos en proyectos de NER. Sube tu archivo, selecciona las técnicas de filtrado y visualiza los resultados."
    ),
    ui.layout_columns(
        ui.value_box(
            "Términos de entrada",
            ui.output_text("n_total"),
            
        ),
        ui.value_box(
            "Conservados",
            ui.output_text("n_conservados"),
            
        ),
        ui.value_box(
            "Eliminados",
            ui.output_text("n_eliminados"),
            
        ),
    ),
    ui.navset_card_tab(
    ui.nav_panel(
            "Conservados",
    ui.output_data_frame("tabla_conservados"),
        ),
        ui.nav_panel(
            "Eliminados",
            ui.output_data_frame("tabla_eliminados"),
        ),
        ui.nav_panel(
            "Resumen por técnica",
            ui.output_data_frame("tabla_resumen"),
        ),
    ),
   
    title="TFG · NER Filter",
)


def server(input, output, session):
    @reactive.calc
    def df_entrada() -> pd.DataFrame:
        f = input.archivo()
        if not f:
            return pd.DataFrame(columns=["término", "frecuencia"])
        path = Path(f[0]["datapath"])
        return parse_input_file(path.read_text(encoding="utf-8"))

    @reactive.calc
    def resultado() -> tuple[pd.DataFrame, pd.DataFrame]:
        df = df_entrada()
        if df.empty:
            empty = pd.DataFrame(columns=["término", "frecuencia"])
            return empty, empty.assign(motivo=pd.Series(dtype=str))
        blacklist_set = {
            line.strip().lower()
            for line in input.blacklist().splitlines()
            if line.strip()
        }
       # Construir tupla de reglas activas leyendo cada checkbox individual
        reglas_activas = tuple(
            rid for rid in REGLAS_DISPONIBLES.keys()
            if input[rid]()
        )
        return filtrar(
            df,
            reglas_activas,
            input.usar_pos(),
            input.usar_blacklist(),
            blacklist_set,
        )

    @reactive.effect
    def _persistir_blacklist():
        """
        Cada vez que el usuario edita la lista negra (al perder el foco),
        se guarda el contenido actual en disco.
        """
        guardar_blacklist(input.blacklist())

    @render.text
    def n_total():
        return f"{len(df_entrada()):,}"

    @render.text
    def n_conservados():
        return f"{len(resultado()[0]):,}"

    @render.text
    def n_eliminados():
        return f"{len(resultado()[1]):,}"

    @render.data_frame
    def tabla_conservados():
        return render.DataGrid(resultado()[0], filters=True, height="70vh",width="100%" )

    @render.data_frame
    def tabla_eliminados():
        return render.DataGrid(resultado()[1], filters=True, height="70vh",width="100%")

    @render.data_frame
    def tabla_resumen():
        eliminados = resultado()[1]
        if eliminados.empty:
            return pd.DataFrame(columns=["técnica", "Eliminados"])
        resumen = (
            eliminados.groupby("motivo", as_index=False)
            .size()
            .rename(columns={"motivo": "técnica", "size": "Eliminados"})
            .sort_values("Eliminados", ascending=False)
            .reset_index(drop=True)
        )
        return render.DataGrid(resumen, height="70vh", width="100%")
   

app = App(app_ui, server)