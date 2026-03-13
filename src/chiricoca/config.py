import matplotlib as mpl
import seaborn as sns
import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd


def setup_style(
    style="whitegrid",
    context="paper",
    dpi=192,
    font_family=None,
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

    if font_family:
        # esto depende de las fuentes que tengas instaladas en el sistema.
        if font_family in mpl.font_manager.get_font_names():
            mpl.rcParams["font.family"] = font_family
    else:
        

        mpl.rcParams["font.family"] = "sans-serif"
        mpl.rcParams["font.sans-serif"] = get_default_font_properties().get_name()

    if formatter_limits:
        mpl.rcParams["axes.formatter.limits"] = (-99, 99)

    pd.set_option("display.max_rows", max_pandas_rows)


def get_default_font_path():
    return Path(os.path.dirname(__file__)) / "assets" / "RobotoCondensed-Regular.ttf"


def get_default_font_properties():
    font_path = get_default_font_path()
    font_manager.fontManager.addfont(font_path)
    return font_manager.FontProperties(fname=font_path)