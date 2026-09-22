import matplotlib
import matplotlib.colors as mcolors
import numpy as np


def truncated_mpl_colormap(cmap_name, min_val=0.1, max_val=0.9, n=128):
    """Recorta un colormap de matplotlib al tramo `[min_val, max_val]` de su rango."""
    cmap = matplotlib.colormaps[cmap_name] if isinstance(cmap_name, str) else cmap_name
    colors = cmap(np.linspace(min_val, max_val, n))
    return mcolors.LinearSegmentedColormap.from_list(f"trunc_{cmap.name}", colors)
