import numpy as np
import pandas as pd


def _get_sort_key(df, sort_by, axis='rows'):
    """
    Calcula la clave de ordenamiento para filas o columnas.
    
    Parameters
    ----------
    df : DataFrame
        Datos a ordenar.
    sort_by : str o list
        Criterio de ordenamiento:
        - 'sum': suma total
        - 'name': alfabético por índice/columna
        - 'entropy': entropía de Shannon (menor = más concentrado)
        - 'gini': coeficiente de Gini (mayor = más desigual)
        - str: nombre de columna/fila específica
        - list: orden explícito (retorna None)
    axis : str
        'rows' o 'columns'.
    
    Returns
    -------
    pd.Series o None
        Clave de ordenamiento, o None si sort_by es una lista.
    """
    if isinstance(sort_by, list):
        return None
    
    if axis == 'rows':
        data = df
        sum_axis = 1
    else:
        data = df.T
        sum_axis = 1
    
    if sort_by == 'sum' or sort_by == True:
        return data.sum(axis=sum_axis)
    elif sort_by == 'name':
        return pd.Series(range(len(data.index)), index=data.index)
    elif sort_by == 'entropy':
        props = data.div(data.sum(axis=sum_axis), axis=0)
        return -(props * np.log(props + 1e-10)).sum(axis=sum_axis)
    elif sort_by == 'gini':
        props = data.div(data.sum(axis=sum_axis), axis=0).values
        n = props.shape[1]
        sorted_props = np.sort(props, axis=1)
        index = np.arange(1, n + 1)
        return pd.Series(((2 * index - n - 1) * sorted_props).sum(axis=1) / n, index=data.index)
    elif isinstance(sort_by, str):
        return data[sort_by] if axis == 'rows' else df.loc[sort_by]
    else:
        return data.sum(axis=sum_axis)