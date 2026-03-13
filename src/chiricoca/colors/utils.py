import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

def truncated_mpl_colormap(cmap_name, min_val=0.1, max_val=0.9, n=128):
    cmap = plt.cm.get_cmap(cmap_name)
    colors = cmap(np.linspace(min_val, max_val, n))
    return mcolors.LinearSegmentedColormap.from_list(
        f'trunc_{cmap_name}', colors
    )
