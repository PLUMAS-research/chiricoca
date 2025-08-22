import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import minmax_scale, normalize, quantile_transform
import numpy as np


def normalize_rows(df):
    return df.div(df.sum(axis=1), axis=0)


def normalize_columns(df):
    return normalize_rows(df.T).T


def standardize_columns(df):
    return df.apply(lambda x: (x - x.mean()) / x.std(), axis=0)


def standardize_rows(df):
    return df.apply(lambda x: (x - x.mean()) / x.std(), axis=1)


def minmax_columns(df):
    return pd.DataFrame(minmax_scale(df, axis=0), index=df.index, columns=df.columns)


def quantile_transform_columns(df, n_quantiles=10, output_distribution="uniform"):
    return pd.DataFrame(
        quantile_transform(
            df,
            axis=0,
            n_quantiles=n_quantiles,
            output_distribution=output_distribution,
            copy=True,
        ),
        index=df.index,
        columns=df.columns,
    )


def weighted_mean(df, value_column, weighs_column):
    weighted_sum = (df[value_column] * df[weighs_column]).sum()
    return weighted_sum / df[weighs_column].sum()


def variance_stabilization(df, smoothing=0.5):
    """
    Calcula pesos para términos usando la transformación arcoseno
    con la fórmula arcsin(p) - arcsin(not p)

    Parámetros:
    df: pandas DataFrame donde filas son documentos y columnas son términos
    smoothing: parámetro de suavizado para evitar valores extremos

    Retorna:
    DataFrame con pesos de términos transformados
    """
    # Inicializar DataFrame de resultados
    weights = pd.DataFrame(index=df.index, columns=df.columns)

    for col in df.columns:
        # Denominador ajustado para que p + not_p = 1
        denominador = df.sum(axis=1) + 2 * smoothing

        # Proporción del término actual (con smoothing)
        p = (df[col] + smoothing) / denominador

        # Proporción de todos los otros términos (con smoothing)
        not_p = (df.sum(axis=1) - df[col] + smoothing) / denominador

        # Aplicar transformación arcoseno y calcular la diferencia
        weights[col] = np.arcsin(np.sqrt(p)) - np.arcsin(np.sqrt(not_p))

    return weights
