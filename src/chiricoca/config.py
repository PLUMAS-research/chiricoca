import os
import warnings
from importlib import resources
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import font_manager


def register_fonts(pattern):
    """Registra en matplotlib las fuentes que calzan con un patrón de rutas.

    matplotlib solo conoce las fuentes instaladas en el sistema, así que una
    tipografía que vive dentro del directorio de una fundición o junto al
    proyecto no se puede usar solo nombrándola. Acepta un patrón estilo glob,
    con `~` expandido, y devuelve las familias que quedaron disponibles.

    >>> register_fonts("~/.fonts/Latinotype*/**/*.otf")
    """
    familias = set()
    pattern = Path(pattern).expanduser()
    raiz = Path(pattern.anchor or ".")
    for ruta in raiz.glob(str(pattern.relative_to(pattern.anchor))):
        try:
            font_manager.fontManager.addfont(ruta)
            familias.add(font_manager.FontProperties(fname=ruta).get_name())
        except (RuntimeError, OSError):
            continue
    return sorted(familias)


def number_format(ax=None, axis="both", decimals=0, locale="es"):
    """Aplica el formato de miles y decimales del idioma a los ejes.

    En castellano el separador de miles es el punto y el decimal es la coma, al
    revés que el formato por omisión de matplotlib. Un eje que muestra 63,627
    donde debería decir 63.627 no es un problema de estilo: dice otro número.
    """
    ax = ax if ax is not None else plt.gca()
    miles, decimal = (".", ",") if locale == "es" else (",", ".")

    def formatear(valor, _=None):
        texto = f"{valor:,.{decimals}f}"
        return texto.replace(",", "\x00").replace(".", decimal).replace("\x00", miles)

    ejes = {"x": [ax.xaxis], "y": [ax.yaxis], "both": [ax.xaxis, ax.yaxis]}[axis]
    for eje in ejes:
        eje.set_major_formatter(formatear)
    return ax


def setup_style(
    style="whitegrid",
    context="paper",
    dpi=192,
    font_family=None,
    font_glob=None,
    formatter_limits=True,
    font_scale=0.9,
    palette="plasma",
    max_pandas_rows=200,
):
    sns.set_style(style)
    sns.set_context(context, font_scale=font_scale)
    # esto configura la calidad de la imagen. dependerá de tu resolución. el valor por omisión es 80
    mpl.rcParams["figure.dpi"] = dpi
    mpl.rcParams["figure.constrained_layout.use"] = True

    mpl.rcParams["scatter.marker"] = "."

    if font_glob:
        # la tipografía puede no estar instalada en el sistema: se registra
        # antes de nombrarla, si no matplotlib cae en silencio a la por omisión.
        register_fonts(font_glob)

    if font_family:
        if font_family in mpl.font_manager.get_font_names():
            mpl.rcParams["font.family"] = font_family
        else:
            warnings.warn(
                f"La familia tipográfica '{font_family}' no está disponible y se "
                "usará la por omisión. Si el archivo existe pero no está instalado, "
                "pasar su ruta en font_glob.",
                stacklevel=2,
            )
    else:
        mpl.rcParams["font.family"] = "sans-serif"
        mpl.rcParams["font.sans-serif"] = get_default_font_properties().get_name()

    if formatter_limits:
        mpl.rcParams["axes.formatter.limits"] = (-99, 99)

    pd.set_option("display.max_rows", max_pandas_rows)


def get_default_font_path():
    return resources.files("chiricoca.assets").joinpath("RobotoCondensed-Regular.ttf")


def get_default_font_properties():
    font_path = get_default_font_path()
    font_manager.fontManager.addfont(font_path)
    return font_manager.FontProperties(fname=font_path)
