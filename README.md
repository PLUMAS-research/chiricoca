# `chiricoca`

Módulo de Python para visualización y análisis geoespacial, construido sobre `matplotlib`. Se usa en los cursos CC5208 (Visualización de Información) y CC5216 (Ciencia de Datos Geográficos) del DCC de la Universidad de Chile.

## Instalación

El paquete no está en PyPI. Para instalarlo con `uv`:

```bash
uv add git+https://github.com/PLUMAS-research/chiricoca.git
```

Requiere Python >= 3.11.

## Módulos

### `config`
Configura los valores por defecto de `matplotlib`: fuente, DPI y despine.

```python
from chiricoca.config import setup_style
setup_style(dpi=150, font_size=11)
```

### `maps`
Mapas coropléticos, de burbujas, de puntos, de calor y LISA.

```python
from chiricoca.maps import choropleth_map, bubble_map, compute_lisa, lisa_map

choropleth_map(geodf, column="poblacion", k=5, binning="fisher_jenks", palette="YlOrRd")
bubble_map(geodf, size="valor", scale=100)

geodf_lisa, moran = compute_lisa(geodf, column="ingreso")
lisa_map(geodf_lisa, ax=ax, title="Clusters LISA")
```

### `geo`
Utilidades geoespaciales: grillas H3, recorte de geometrías, coloreado de polígonos, KDE y matrices de distancia.

```python
from chiricoca.geo.grid import h3_grid_from_bounds
from chiricoca.geo.utils import k_coloreo, clip_area_geodataframe
from chiricoca.geo.figures import figure_from_geodataframe
```

### `tables`
Gráficos estadísticos: Marimekko, streamgraph, barras apiladas, burbujas, scatter, heatmap y ternario.

```python
from chiricoca.tables import marimekko, streamgraph, barchart, scatterplot
```

### `colors`
Leyendas de color categóricas y continuas, colormaps desde paletas de seaborn y matrices bivariadas.

```python
from chiricoca.colors import categorical_color_legend, add_ranged_color_legend
```

### `base`
Ponderación (`normalize_rows`, `tfidf`, `weighted_mean`), algoritmos de posicionamiento de etiquetas y utilidades de ordenamiento.

### `networks`
Estadísticas de redes.

```python
from chiricoca.networks.stats import summary
summary(G, root_node="A")
```

### `text`
Nubes de palabras con coloreado por tamaño.

## Dependencias

Las dependencias se gestionan con `uv` y están declaradas en `pyproject.toml`. Para instalarlas: `uv sync`.
