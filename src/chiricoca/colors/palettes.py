import matplotlib.colors as colors
import numpy as np
import seaborn as sns


def sample_palette(source, n):
    """`n` colores de una paleta de seaborn (por nombre) o de un `Colormap` de matplotlib."""
    if isinstance(source, colors.Colormap):
        return [source(v) for v in np.linspace(0, 1, n)]
    return sns.color_palette(source, n_colors=n)


def build_palette(
    bins,
    palette=None,
    center_value=None,
    default_divergent="RdBu_r",
    default_negative="Blues_r",
    default_positive="Reds",
    palette_type="light",
):
    """Un color por clase para los bordes `bins`, divergente si las clases cruzan el centro.

    Parameters
    ----------
    bins : array-like
        Bordes de las clases, del mínimo al máximo; hay `len(bins) - 1`
        clases.
    palette : str, color o Colormap, optional
        Un nombre de paleta de seaborn o un `Colormap` se muestrea con una
        color por clase (o se centra, ver abajo). Un color solo produce una
        rampa clara u oscura según `palette_type`, nunca divergente. Sin
        paleta se usa `default_divergent` si las clases cruzan el centro,
        `default_negative` si están todas por debajo y `default_positive` si
        están por encima.
    center_value : float, optional
        Valor en que se centra la paleta divergente; cero por omisión.

    Notes
    -----
    Con clases a ambos lados del centro, la clase que lo contiene recibe el
    color central de la paleta. Si hay más clases a un lado que al otro, se
    muestrea una paleta más larga y se descarta el extremo sobrante, así los
    colores siguen simétricos respecto del centro aunque las clases no lo
    sean.
    """
    bins = np.asarray(bins, dtype=float)
    k = len(bins) - 1
    midpoint = center_value if center_value is not None else 0.0
    divergente = bins[0] < midpoint < bins[-1]

    if palette is None:
        if divergente:
            source = default_divergent
        elif bins[-1] <= midpoint:
            source = default_negative
        else:
            source = default_positive
    elif not isinstance(palette, colors.Colormap) and colors.is_color_like(palette):
        constructor = sns.light_palette if palette_type == "light" else sns.dark_palette
        return list(constructor(palette, n_colors=k))
    else:
        source = palette

    if not divergente:
        return list(sample_palette(source, k))

    middle_idx = int(np.clip(np.searchsorted(bins, midpoint, side="right") - 1, 0, k - 1))
    left, right = middle_idx, k - middle_idx - 1
    delta = abs(left - right)
    start_idx = max(0, right - left)
    return list(sample_palette(source, k + delta)[start_idx : start_idx + k])
