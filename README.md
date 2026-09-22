# `chiricoca`

Módulo de Python para visualización y análisis geoespacial, construido sobre `matplotlib`. Se usa en los cursos CC5208 (Visualización de Información) y CC5216 (Ciencia de Datos Geográficos) del DCC de la Universidad de Chile, y en los proyectos del grupo PLUMAS (PLataforma Urbana de Movilidad, Análisis y Simulación).

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
Mapas coropléticos, de burbujas, de puntos, de calor, LISA, de flujos, cartogramas e isolíneas.

```python
from chiricoca.maps import choropleth_map, bubble_map, compute_lisa, lisa_map

choropleth_map(geodf, column="poblacion", k=5, binning="fisher_jenks", palette="YlOrRd")
bubble_map(geodf, size="valor", scale=100)

geodf_lisa, moran = compute_lisa(geodf, column="ingreso")
lisa_map(geodf_lisa, ax=ax, title="Clusters LISA")
```

Los mapas de flujo reciben una matriz origen-destino y la geometría de los lugares. Con todos los pares el mapa es una maraña, así que `flow_map` combina un filtro (`disparity`, `top_k`, `share`, `min_weight`), la agregación de sentidos (`aggregate`) y el agrupamiento en haces (`bundle`) para dar un panorama de la red.

```python
from chiricoca.maps import flow_map, flow_tree, line_heat_map, cartogram_map, contour_map

ax, dibujados = flow_map(od, comunas, "origen", "destino", "viajes", node_column="comuna",
                         disparity=0.05, bundle=True, arrows=True, node_scale=0.02,
                         legend=True, return_flows=True)
flow_tree(od, comunas, "origen", "destino", "viajes", origin="Puente Alto", node_column="comuna")
line_heat_map(recorridos, weight="buses_hora", segment_length=60, snap=30)
ax, deformado = cartogram_map(comunas, "poblacion", kind="dorling")
```

### `geo`
Utilidades geoespaciales: grillas H3, conteo de puntos por celda, recorte de geometrías, coloreado de polígonos, KDE y matrices de distancia.

```python
from chiricoca.geo.grid import h3_grid_from_bounds, count_in_grid
from chiricoca.geo.utils import k_coloreo, clip_area_geodataframe, projected_centroids
from chiricoca.geo.figures import figure_from_geodataframe
```

### `tables`
Gráficos estadísticos: Marimekko, streamgraph, barras apiladas, burbujas, scatter, slopegraph, heatmap con clustering jerárquico, ternario y coordenadas paralelas.

```python
from chiricoca.tables import marimekko, streamgraph, barchart, scatterplot, slopegraph, parallel_coordinates
```

En los gráficos apilados verticales (`barchart`, `marimekko`, `streamgraph`), `legend="right"` rotula cada serie a la altura en que termina, con una línea guía hasta el rótulo.

### `colors`
Leyendas de color categóricas y continuas, construcción de paletas por clases, colormaps desde paletas de seaborn y matrices bivariadas. `BIVARIATE_BLUE_ORANGE` y `BIVARIATE_PURPLE_TEAL` son dos matrices bivariadas de 3 por 3 validadas contra deuteranopia y protanopia, con luminancia decreciente en ambos ejes.

```python
from chiricoca.colors import (
    categorical_color_legend,
    add_ranged_color_legend,
    build_palette,
    BIVARIATE_BLUE_ORANGE,
)
```

### `base`
Ponderación (`normalize_rows`, `tfidf`, `weighted_mean`), algoritmos de posicionamiento de etiquetas, utilidades de ordenamiento y resolución de raíz de proyecto (`add_project_root`) para bootstrap de scripts en subcarpetas.

```python
from chiricoca.base.paths import add_project_root
add_project_root()
```

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
