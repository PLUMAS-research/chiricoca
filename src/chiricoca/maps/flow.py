import geopandas as gpd
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from chiricoca.colors import categorical_color_legend, categorical_colors, colormap_from_palette
from chiricoca.geo.figures import figure_from_geodataframe
from chiricoca.geo.utils import projected_centroids


def curved_line(origin, destination, curvature=0.2, n_points=48):
    """Curva de Bézier cuadrática entre dos puntos, como arreglo de coordenadas.

    El punto de control se desplaza perpendicular a la recta que une los dos
    extremos, a una distancia proporcional a su largo. Con eso la ida y la
    vuelta de un mismo par no quedan una encima de la otra, que es lo que pasa
    al unirlos con una recta.

    Parameters
    ----------
    origin, destination : tuple of float
        Coordenadas de los dos extremos.
    curvature : float
        Desplazamiento del punto de control, como fracción del largo de la
        recta. Con 0 devuelve la recta.
    n_points : int
        Cuántos puntos tiene la curva resultante.
    """
    x0, y0 = origin
    x1, y1 = destination
    dx, dy = x1 - x0, y1 - y0
    length = np.hypot(dx, dy)

    if length == 0 or curvature == 0:
        return np.column_stack(
            [np.linspace(x0, x1, n_points), np.linspace(y0, y1, n_points)]
        )

    # normal unitaria a la recta
    px, py = -dy / length, dx / length
    control = np.array(
        [(x0 + x1) / 2 + length * curvature * px, (y0 + y1) / 2 + length * curvature * py]
    )

    t = np.linspace(0, 1, n_points)[:, None]
    p0 = np.array([x0, y0])
    p2 = np.array([x1, y1])
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * control + t**2 * p2


# ---------------------------------------------------------------------------
# Qué flujos mostrar
# ---------------------------------------------------------------------------


def filter_flows(
    flows,
    source,
    target,
    weight,
    min_weight=None,
    top_k=None,
    top_k_in=None,
    local_share=None,
    share=None,
    disparity=None,
    surplus=None,
    spanning_tree=False,
):
    """Selecciona los flujos que vale la pena dibujar.

    Una matriz origen-destino completa tiene más pares de los que caben en un
    mapa: con cuarenta lugares son mil seiscientos, y dibujarlos todos da una
    maraña donde no se lee nada. Los criterios se acumulan: un flujo sobrevive
    si pasa todos los que se entregan. Los dos criterios relativos a una
    distribución (`disparity`, `surplus`) se evalúan primero, sobre la matriz
    completa, porque dependen de cuántos flujos tiene cada lugar y de cómo se
    reparte su total; los demás se aplican después, en el orden `min_weight`,
    `top_k`, `top_k_in`, `local_share`, `share`. La excepción es
    `spanning_tree`, que agrega flujos en vez de quitarlos.

    Parameters
    ----------
    flows : DataFrame
        Una fila por par, con las columnas `source`, `target` y `weight`.
    min_weight : float, optional
        Descarta los flujos con peso menor. Es el criterio más simple y el más
        sesgado: conserva lo que sale de los lugares grandes y borra del mapa a
        los chicos.
    top_k : int, optional
        Conserva los `top_k` flujos de mayor peso que salen de cada origen. Así
        cada lugar mantiene sus destinos principales, tenga mucho o poco flujo,
        y la estructura local sobrevive.
    top_k_in : int, optional
        Lo mismo para los flujos que llegan a cada destino. Con `top_k` y
        `top_k_in` juntos, un flujo tiene que ser principal para su origen y
        para su destino.
    local_share : float, optional
        Conserva, para cada origen, sus flujos más grandes hasta acumular esta
        fracción de lo que sale de él, entre 0 y 1. Es `top_k` con un k que se
        adapta a cuán concentrado está cada lugar: un lugar que manda todo a un
        destino conserva un flujo, uno que reparte conserva varios.
    share : float, optional
        Conserva los flujos más grandes de toda la matriz hasta acumular esta
        fracción del peso total, entre 0 y 1. Dice cuánto del fenómeno queda
        representado.
    disparity : float, optional
        Nivel de significancia del filtro de disparidad de Serrano, Boguñá y
        Vespignani (2009). Para un lugar con `k` flujos, un flujo que lleva la
        fracción `p` de su total tiene significancia `(1 - p) ** (k - 1)`, y se
        conserva si ese valor es menor que `disparity` respecto de lo que sale
        de su origen o de lo que llega a su destino. Extrae la columna vertebral
        de la red: los flujos que pesan más de lo que pesarían si cada lugar
        repartiera su total al azar entre sus vecinos. Un lugar con un solo
        flujo nunca lo conserva por sí mismo. Valores usuales: 0,05 o 0,01.
    surplus : float, optional
        Conserva los flujos que superan en este factor lo que se esperaría si
        origen y destino fueran independientes: el esperado del par es lo que
        sale del origen por lo que llega al destino, dividido por el total de
        la matriz (el mismo esperado de una tabla de contingencia). Con 1 quedan
        los pares con más flujo del esperado; con 2, los que lo duplican. A
        diferencia de `disparity`, un flujo grande entre dos lugares grandes
        no es notable por sí solo: lo es la afinidad entre ambos.
    spanning_tree : bool
        Agrega el árbol de expansión máximo de la red no dirigida (con la suma
        de ambos sentidos como peso), así ningún lugar queda desconectado del
        mapa aunque los demás criterios hayan descartado todos sus flujos. Se
        conservan los dos sentidos de cada par del árbol que existan en
        `flows`.

    Returns
    -------
    DataFrame
        Las filas de `flows` que pasan los criterios.
    """
    pairs = flows.reset_index(drop=True)
    pesos = pairs[weight].to_numpy(dtype=float)
    conserva = np.ones(len(pairs), dtype=bool)

    def total_por(columna):
        return pairs.groupby(columna, observed=True)[weight].transform("sum").to_numpy()

    if disparity is not None:

        def significancia(columna):
            grupo = pairs.groupby(columna, observed=True)[weight]
            p = pesos / grupo.transform("sum").to_numpy()
            return (1 - p) ** (grupo.transform("size").to_numpy() - 1)

        conserva &= (significancia(source) < disparity) | (significancia(target) < disparity)

    if surplus is not None:
        esperado = total_por(source) * total_por(target) / pesos.sum()
        conserva &= pesos >= surplus * esperado

    if min_weight is not None:
        conserva &= pesos >= min_weight

    def por_grupo(columna, criterio):
        """Aplica `criterio` a los flujos que quedan, ordenados de mayor a menor, agrupados por `columna`."""
        vivos = pairs[conserva].sort_values(weight, ascending=False)
        return conserva & pairs.index.isin(criterio(vivos, vivos.groupby(columna, observed=True, sort=False)))

    if top_k is not None:
        conserva = por_grupo(source, lambda vivos, grupo: grupo.head(top_k).index)
    if top_k_in is not None:
        conserva = por_grupo(target, lambda vivos, grupo: grupo.head(top_k_in).index)
    if local_share is not None:
        # queda cada flujo mientras lo acumulado antes de él no llegue a la fracción
        conserva = por_grupo(
            source,
            lambda vivos, grupo: vivos.index[
                (grupo[weight].cumsum() - vivos[weight]) / grupo[weight].transform("sum") < local_share
            ],
        )
    if share is not None:
        vivos = pairs[conserva].sort_values(weight, ascending=False)
        acumulado = vivos[weight].cumsum() / vivos[weight].sum()
        conserva &= pairs.index.isin(vivos.index[acumulado.shift(fill_value=0) < share])

    if spanning_tree:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import minimum_spanning_tree

        lugares = pd.Index(pd.unique(np.concatenate([pairs[source].to_numpy(), pairs[target].to_numpy()])))
        i = lugares.get_indexer(pairs[source])
        j = lugares.get_indexer(pairs[target])
        lo, hi = np.minimum(i, j), np.maximum(i, j)
        distinto = lo != hi
        simetrico = pd.DataFrame({"a": lo[distinto], "b": hi[distinto], "w": pesos[distinto]}).groupby(["a", "b"], as_index=False).sum()
        # el árbol de expansión mínimo sobre pesos invertidos es el máximo sobre los originales
        invertido = simetrico["w"].max() + 1 - simetrico["w"]
        matriz = coo_matrix((invertido, (simetrico["a"], simetrico["b"])), shape=(len(lugares), len(lugares)))
        arbol = minimum_spanning_tree(matriz).tocoo()
        en_arbol = np.minimum(arbol.row, arbol.col) * len(lugares) + np.maximum(arbol.row, arbol.col)
        conserva |= distinto & np.isin(lo * len(lugares) + hi, en_arbol)

    return flows[conserva]


def edge_triangles(flows, source, target):
    """Cantidad de triángulos en que participa cada flujo, sobre la red no dirigida.

    Un flujo está en un triángulo cuando sus dos extremos comparten un tercer
    lugar conectado con ambos. Separa los flujos que forman circuitos locales
    de los que solo unen dos lugares: en las redes de cuidado dominan los
    primeros y en las de empleo los segundos. Se calcula como el producto de la
    matriz de adyacencia consigo misma, leído en las aristas.

    Returns
    -------
    Series
        Alineada con `flows`; `> 0` da la clase "en triángulo". Los flujos de
        un lugar consigo mismo valen 0.
    """
    from scipy.sparse import coo_matrix

    a, b = flows[source].to_numpy(), flows[target].to_numpy()
    lugares = pd.Index(pd.unique(np.concatenate([a, b])))
    i, j = lugares.get_indexer(a), lugares.get_indexer(b)
    distinto = i != j
    n = len(lugares)
    adyacencia = coo_matrix((np.ones(distinto.sum()), (i[distinto], j[distinto])), shape=(n, n)).tocsr()
    adyacencia = ((adyacencia + adyacencia.T) > 0).astype(np.int64)
    comunes = np.asarray((adyacencia @ adyacencia)[i, j]).ravel()
    return pd.Series(np.where(distinto, comunes, 0), index=flows.index)


def aggregate_flows(flows, source, target, weight, how="total"):
    """Combina la ida y la vuelta de cada par en un solo flujo.

    Parameters
    ----------
    how : {"total", "net"}
        Con `"total"` suma los dos sentidos; el resultado va del identificador
        menor al mayor y no tiene dirección. Con `"net"` resta la vuelta a la
        ida y deja el saldo en el sentido dominante, así el mapa muestra hacia
        dónde se mueve el balance; los pares equilibrados desaparecen.

    Returns
    -------
    DataFrame
        Con las columnas `source`, `target` y `weight`. Las demás columnas de
        `flows` no sobreviven, porque no tienen un valor definido para el par.
    """
    if how not in ("total", "net"):
        raise ValueError(f"how debe ser 'total' o 'net', no {how!r}")

    a, b = flows[source].to_numpy(), flows[target].to_numpy()
    directo = a <= b
    pesos = flows[weight].to_numpy(dtype=float)
    tabla = pd.DataFrame(
        {
            source: np.where(directo, a, b),
            target: np.where(directo, b, a),
            "ida": np.where(directo, pesos, 0.0),
            "vuelta": np.where(directo, 0.0, pesos),
        }
    )
    pares = tabla.groupby([source, target], as_index=False, sort=False).sum()

    if how == "total":
        pares[weight] = pares["ida"] + pares["vuelta"]
    else:
        saldo = pares["ida"] - pares["vuelta"]
        invertir = saldo < 0
        pares.loc[invertir, [source, target]] = pares.loc[invertir, [target, source]].to_numpy()
        pares[weight] = saldo.abs()
        pares = pares[pares[weight] > 0]

    return pares[[source, target, weight]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Bundling
# ---------------------------------------------------------------------------


def _lengths(paths):
    """Largo de arco de cada trazado de `paths` (E, m, 2)."""
    return np.hypot(*np.diff(paths, axis=1).transpose(2, 0, 1)).sum(axis=1)


def _interpolate(paths, fractions):
    """Puntos de cada trazado en las fracciones de su largo de arco pedidas.

    `paths` es (E, m, 2) y `fractions` (E, k), con valores en [0, 1]. Devuelve
    (E, k, 2). Todas las filas se resuelven con una sola búsqueda binaria: la
    fracción acumulada de cada fila se desplaza dos unidades por fila, así el
    arreglo aplanado queda ordenado y `searchsorted` ubica de una vez el
    segmento que contiene cada fracción.
    """
    E, m, _ = paths.shape
    saltos = np.hypot(*np.diff(paths, axis=1).transpose(2, 0, 1))
    acumulado = np.concatenate([np.zeros((E, 1)), saltos.cumsum(axis=1)], axis=1)
    total = acumulado[:, -1:]
    acumulado = acumulado / np.where(total == 0, 1, total)

    fila = 2.0 * np.arange(E)[:, None]
    indice = np.searchsorted((acumulado + fila).ravel(), (fractions + fila).ravel(), side="right") - 1
    indice = np.clip(indice.reshape(E, -1) - (np.arange(E) * m)[:, None], 0, m - 2)
    filas = np.arange(E)[:, None]
    s0, s1 = acumulado[filas, indice], acumulado[filas, indice + 1]
    t = np.clip((fractions - s0) / np.where(s1 > s0, s1 - s0, 1), 0, 1)[..., None]
    return paths[filas, indice] * (1 - t) + paths[filas, indice + 1] * t


def _resample(paths, n):
    """Remuestrea cada trazado de `paths` (E, m, 2) a `n` puntos equiespaciados."""
    paths = np.asarray(paths, dtype=float)
    return _interpolate(paths, np.broadcast_to(np.linspace(0, 1, n), (len(paths), n)))


def bundle_flows(
    paths,
    weights=None,
    iterations=10,
    bandwidth=0.1,
    decay=0.7,
    step=1.0,
    n_points=64,
    resolution=400,
    smooth=1.5,
):
    """Agrupa en haces los trazados que comparten corredor, por densidad.

    Es el método de Hurter, Ersoy y Telea (2012): los trazados se muestrean
    en puntos, se estima la densidad de puntos sobre una grilla con un kernel
    gaussiano, y cada punto avanza en la dirección en que crece la densidad, o
    sea hacia el corredor más cargado que tiene cerca. El ancho del kernel se
    reduce en cada iteración, así primero se forman los corredores gruesos y
    después se afinan. Los extremos no se mueven.

    Agrupa por cercanía y no por parecido entre trazados, así que un flujo
    corto y uno largo que salen del mismo lugar hacia el mismo lado terminan en
    el mismo haz. Eso es lo que pasa en una matriz origen-destino, donde casi
    todo sale de unos pocos centros hacia destinos a distancias muy distintas.
    El costo es el de cualquier bundling: un haz dice entre qué zonas se mueve
    el flujo, no por dónde pasa cada uno.

    Parameters
    ----------
    paths : sequence of (n, 2) arrays
        Trazados. Se remuestrean a `n_points` puntos equiespaciados.
    weights : array-like, optional
        Peso de cada trazado en la densidad. Con pesos, los flujos chicos se
        acercan a los corredores de los grandes; sin ellos todos pesan igual.
    iterations : int
        Cuántas veces se estima la densidad y se mueven los puntos.
    bandwidth : float
        Ancho inicial del kernel, como fracción del lado mayor de la caja que
        contiene los trazados. Decide qué tan lejos se buscan compañeros de
        corredor: con 0,05 solo se juntan trazados casi superpuestos; con 0,2
        el mapa se reduce a unos pocos troncos.
    decay : float
        Factor por el que se multiplica el ancho del kernel en cada iteración.
    step : float
        Cuánto avanza por iteración el punto donde el gradiente es máximo, en
        anchos de kernel. Los demás avanzan en proporción a su gradiente.
    n_points : int
        Puntos por trazado.
    resolution : int
        Celdas por lado de la grilla de densidad.
    smooth : float
        Sigma del suavizado gaussiano final a lo largo de cada trazado, en
        puntos. Con 0 no se suaviza.

    Returns
    -------
    list of ndarray
        Un trazado por elemento de `paths`, con `n_points` puntos.

    Notes
    -----
    El costo es lineal en la cantidad de puntos, así que sirve para decenas de
    miles de trazados. Las coordenadas se normalizan a una caja de mil
    unidades antes de iterar y `bandwidth` se refiere a esa caja, con lo que
    el resultado no depende de la escala del mapa.
    """
    from scipy.ndimage import gaussian_filter, gaussian_filter1d, map_coordinates

    puntos = [np.asarray(p, dtype=float) for p in paths]
    if not puntos:
        return []
    if len({len(p) for p in puntos}) == 1:
        X = _resample(np.stack(puntos), n_points)
    else:
        X = np.stack([_resample(p[None, :, :], n_points)[0] for p in puntos])

    todos = X.reshape(-1, 2)
    minimo = todos.min(axis=0)
    extension = (todos.max(axis=0) - minimo).max()
    escala = 1000.0 / extension if extension > 0 else 1.0
    X = (X - minimo) * escala

    # cada punto aporta a la densidad según el peso de su trazado y el largo
    # que representa, así un trazado largo no pesa menos por punto que uno corto
    n_trazados = len(X)
    largos = np.hypot(*np.diff(X, axis=1).transpose(2, 0, 1)).sum(axis=1)
    por_trazado = np.ones(n_trazados) if weights is None else np.asarray(weights, dtype=float)
    por_trazado = por_trazado / max(por_trazado.max(), 1e-12)
    por_punto = np.repeat(por_trazado * largos / n_points, n_points)

    margen = 100.0
    celda = (1000 + 2 * margen) / resolution
    ancho = bandwidth * 1000
    for _ in range(iterations):
        celdas_xy = np.clip(((X.reshape(-1, 2) + margen) / celda).astype(np.intp), 0, resolution - 1)
        densidad = np.bincount(
            celdas_xy[:, 0] * resolution + celdas_xy[:, 1], weights=por_punto, minlength=resolution**2
        ).reshape(resolution, resolution)
        densidad = gaussian_filter(densidad, sigma=ancho / 2 / celda)
        gx, gy = np.gradient(densidad)
        maximo = max(np.hypot(gx, gy).max(), 1e-12)

        celdas = ((X[:, 1:-1].reshape(-1, 2) + margen) / celda - 0.5).T
        empuje = np.column_stack(
            [map_coordinates(gx, celdas, order=1), map_coordinates(gy, celdas, order=1)]
        )
        X[:, 1:-1] += (empuje / maximo * step * ancho).reshape(n_trazados, n_points - 2, 2)

        # un par de pasadas de suavizado laplaciano quitan el serrucho que deja
        # el gradiente muestreado, y el remuestreo evita que los puntos se
        # amontonen dentro del haz
        for _ in range(2):
            X[:, 1:-1] = 0.5 * X[:, 1:-1] + 0.25 * (X[:, :-2] + X[:, 2:])
        X = _resample(X, n_points)
        ancho *= decay

    if smooth > 0:
        X[:, 1:-1] = gaussian_filter1d(X, smooth, axis=1, mode="nearest")[:, 1:-1]

    return list(X / escala + minimo)


# ---------------------------------------------------------------------------
# Geometría en pantalla
# ---------------------------------------------------------------------------


def _screen_transforms(ax):
    """Funciones de datos a píxeles y de vuelta, con el aspecto del eje ya aplicado.

    La geometría de los flujos (curvatura, ancho, puntas, haces) se calcula en
    píxeles y no en unidades del mapa: en un CRS geográfico un grado de latitud
    no mide lo mismo en pantalla que uno de longitud, y en unidades del mapa
    las perpendiculares dejan de serlo. Los límites del eje deben estar fijos
    antes de llamar, porque el resultado queda en coordenadas de datos.
    """
    ax.apply_aspect()
    return ax.transData.transform, ax.transData.inverted().transform


def _points_to_pixels(ax, points):
    return np.asarray(points) * ax.figure.dpi / 72


def _unit(vectors):
    norma = np.hypot(vectors[..., 0], vectors[..., 1])
    return vectors / np.where(norma > 0, norma, 1)[..., None]


def _right_normals(paths):
    """Normal unitaria hacia la derecha de la dirección local de cada trazado (E, n, 2)."""
    delta = np.gradient(paths, axis=1)
    return _unit(np.stack([delta[..., 1], -delta[..., 0]], axis=-1))


def _offset_right(paths, distances):
    """Desplaza cada trazado hacia su derecha la distancia de su fila."""
    return paths + _right_normals(paths) * distances[:, None, None]


def _bezier_curves(starts, ends, curvatures, n_points=48):
    """Curvas de Bézier cuadráticas entre pares de puntos, como arreglo (E, n, 2).

    Es `curved_line` para todos los pares a la vez: el punto de control de cada
    curva se desplaza hacia la izquierda de la recta `curvatures` veces su
    largo. Con curvatura 0 la curva es la recta.
    """
    vector = ends - starts
    largo = np.hypot(vector[:, 0], vector[:, 1])
    normal = _unit(np.stack([-vector[:, 1], vector[:, 0]], axis=-1))
    control = (starts + ends) / 2 + normal * (largo * curvatures)[:, None]
    t = np.linspace(0, 1, n_points)[None, :, None]
    return (1 - t) ** 2 * starts[:, None] + 2 * (1 - t) * t * control[:, None] + t**2 * ends[:, None]


def _shorten(paths, start, end, keep=0.4):
    """Acorta cada trazado `start` unidades al comienzo y `end` al final.

    Deja al menos la fracción `keep` del largo y devuelve los trazados
    remuestreados con la misma cantidad de puntos.
    """
    total = np.maximum(_lengths(paths), 1e-12)
    exceso = (start + end) / total
    factor = np.where(exceso > 1 - keep, (1 - keep) / np.maximum(exceso, 1e-12), 1.0)
    desde = start * factor / total
    hasta = 1 - end * factor / total
    fracciones = desde[:, None] + (hasta - desde)[:, None] * np.linspace(0, 1, paths.shape[1])[None, :]
    return _interpolate(paths, fracciones)


def _cut_heads(paths, lengths):
    """Separa cada trazado en el cuerpo, remuestreado, y la punta final de largo `lengths`.

    Devuelve el cuerpo (E, n, 2), la base de la punta (E, 2) y la dirección
    unitaria de llegada (E, 2). La punta nunca ocupa más del 60% del trazado.
    """
    total = np.maximum(_lengths(paths), 1e-12)
    largo = np.minimum(lengths, 0.6 * total)
    fracciones = (1 - largo / total)[:, None] * np.linspace(0, 1, paths.shape[1])[None, :]
    cuerpo = _interpolate(paths, fracciones)
    base = cuerpo[:, -1]
    return cuerpo, base, _unit(paths[:, -1] - base)


def _tapered_polygons(paths, widths, taper):
    """Convierte cada trazado en un polígono que se afina hacia el final.

    El ancho vale `widths` en el origen y `widths * (1 - taper)` en el
    destino, así la forma indica la dirección sin punta de flecha.
    """
    normal = -_right_normals(paths)
    t = np.linspace(0, 1, paths.shape[1])[None, :]
    medio_ancho = (widths[:, None] / 2 * (1 - taper * t))[..., None]
    return np.concatenate([paths + normal * medio_ancho, (paths - normal * medio_ancho)[:, ::-1]], axis=1)


def _arrowheads(paths, lengths, half_widths):
    """Triángulos (E, 3, 2) en el extremo de cada trazado, y los trazados recortados en su base.

    El trazado se corta en la base porque, si la línea sigue por debajo del
    triángulo, la superposición se nota con transparencia.
    """
    cuerpo, base, direccion = _cut_heads(paths, lengths)
    ala = np.stack([-direccion[:, 1], direccion[:, 0]], axis=-1) * half_widths[:, None]
    cabezas = np.stack([paths[:, -1], base + ala, base - ala], axis=1)
    return cabezas, cuerpo


def _half_arrows(paths, widths, taper, lengths, wings, gap=0.25):
    """Bandas a la derecha de cada línea central, rematadas en media punta de flecha (E, 2n + 2, 2).

    Es la media flecha de Tobler (1987): el flujo ocupa solo el lado derecho de
    la línea que une los dos lugares, así la ida y la vuelta comparten la línea,
    una a cada lado, y el par se lee como una sola flecha doble cuyo ancho total
    es la suma de los dos sentidos. La punta es la mitad de un triángulo: la
    cara interior sigue la línea central y el ala sale hacia afuera. Cada flujo
    es un solo polígono, así la transparencia no marca costuras.
    """
    cuerpo, base, direccion = _cut_heads(paths, lengths)
    derecha_final = np.stack([direccion[:, 1], -direccion[:, 0]], axis=-1)
    derecha = _right_normals(cuerpo)
    # el afinado se mide sobre el largo total, punta incluida
    total = np.maximum(_lengths(paths), 1e-12)
    recorrido = (_lengths(cuerpo) / total)[:, None] * np.linspace(0, 1, cuerpo.shape[1])[None, :]
    ancho = widths[:, None] * (1 - taper * recorrido)
    interior = cuerpo + derecha * gap
    exterior = cuerpo + derecha * (gap + ancho)[..., None]
    extremo = paths[:, -1] + derecha_final * gap
    ala = base + derecha_final * (gap + wings)[:, None]
    return np.concatenate([interior, extremo[:, None], ala[:, None], exterior[:, ::-1]], axis=1)


def _avoiding_curvatures(starts, ends, nodes, clearance, exclude, curvature, cap=0.45, chunk=4096):
    """Curvatura con signo de cada flujo para no pasar sobre los nodos ajenos.

    Un flujo que cruza por encima de un lugar con el que no conecta parece
    detenerse ahí. Para cada nodo cuya proyección cae dentro de la cuerda y que
    queda a menos de `clearance` de ella se calcula cuánto habría que doblar
    hacia cada lado para librarlo (la desviación de una Bézier cuadrática en la
    posición `u` de la cuerda es `2 u (1 - u)` veces el desplazamiento del
    control). Se dobla hacia el lado que exige menos, sin pasar de `cap` y
    nunca menos que la `curvature` pedida. `exclude` (E, 2) trae la posición
    en `nodes` de los dos extremos de cada flujo, que no se esquivan.

    Trabaja con matrices (flujos por nodos), por tandas de `chunk` flujos para
    acotar la memoria.
    """
    resultado = np.full(len(starts), float(curvature))
    for inicio in range(0, len(starts), chunk):
        tramo = slice(inicio, inicio + chunk)
        a, b = starts[tramo], ends[tramo]
        vector = b - a
        largo = np.where(np.hypot(vector[:, 0], vector[:, 1]) > 0, np.hypot(vector[:, 0], vector[:, 1]), 1e-12)
        relativo = nodes[None, :, :] - a[:, None, :]
        u = np.einsum("enk,ek->en", relativo, vector) / largo[:, None] ** 2
        # distancia perpendicular con signo: positiva a la derecha de la cuerda
        lateral = (relativo[..., 0] * vector[:, None, 1] - relativo[..., 1] * vector[:, None, 0]) / largo[:, None]
        holgura = clearance[tramo]
        conflicto = (u > 0.05) & (u < 0.95) & (np.abs(lateral) < holgura)
        conflicto[np.arange(len(a))[:, None], exclude[tramo]] = False

        desviacion = np.where(conflicto, 2 * u * (1 - u) * largo[:, None], 1)
        izquierda = lateral < 0
        lateral = np.abs(lateral)
        # doblar hacia la izquierda aleja los nodos de la derecha y obliga a
        # sobrepasar los de la izquierda; hacia la derecha, al revés
        hacia_izquierda = np.where(conflicto, np.where(izquierda, lateral + holgura, holgura - lateral) / desviacion, -np.inf).max(axis=1)
        hacia_derecha = np.where(conflicto, np.where(izquierda, holgura - lateral, lateral + holgura) / desviacion, -np.inf).max(axis=1)
        hay = conflicto.any(axis=1)
        magnitud_izquierda = np.minimum(np.maximum(hacia_izquierda, abs(curvature)), cap)
        magnitud_derecha = np.minimum(np.maximum(hacia_derecha, abs(curvature)), cap)
        resultado[tramo] = np.where(
            hay, np.where(hacia_izquierda <= hacia_derecha, magnitud_izquierda, -magnitud_derecha), curvature
        )
    return resultado


def _paint(collection, color_values, color, palette, hue_colors, clim, alphas):
    """Colorea una colección por categoría (`hue_colors`), por valor (`palette`) o plana.

    `alphas` es un escalar o un arreglo con la transparencia de cada elemento.
    Con `palette`, el color sigue `color_values` entre los límites `clim`.
    """
    from matplotlib.colors import to_rgba, to_rgba_array

    poligonos = isinstance(collection, PolyCollection)
    pintar = collection.set_facecolor if poligonos else collection.set_color
    por_elemento = np.ndim(alphas) > 0
    if palette is not None and hue_colors is None:
        collection.set_cmap(colormap_from_palette(palette, n_colors=256))
        collection.set_array(np.asarray(color_values, dtype=float))
        collection.set_clim(*clim)
        collection.set_alpha(np.asarray(alphas, dtype=float) if por_elemento else alphas)
    else:
        n = len(collection.get_paths())
        rgba = to_rgba_array(hue_colors) if hue_colors is not None else np.tile(to_rgba(color), (n, 1))
        rgba[:, 3] = alphas
        pintar(rgba)
        collection.set_alpha(None)
    if poligonos:
        collection.set_edgecolor("none")


def _classify(values, binning, k, bins):
    """Clase de cada valor y bordes de las clases, con la convención de `choropleth_map`.

    Las clases son intervalos `(a, b]`, con el mínimo incluido en la primera.
    """
    from mapclassify import FisherJenks, Quantiles

    values = np.asarray(values, dtype=float)
    if binning == "custom":
        if bins is None:
            raise ValueError("width_bins hace falta con width_binning='custom'")
        bordes = np.asarray(bins, dtype=float)
        if len(bordes) < 2 or not np.all(np.diff(bordes) > 0):
            raise ValueError("width_bins debe ser una secuencia creciente de al menos dos bordes")
        if values.min() < bordes[0] or values.max() > bordes[-1]:
            raise ValueError("width_bins no cubre todos los pesos dibujados")
    elif binning == "uniform":
        bordes = np.linspace(values.min(), values.max(), k + 1)
    elif binning in ("quantiles", "fisher_jenks"):
        k = min(k, len(np.unique(values)))
        clasificador = (FisherJenks if binning == "fisher_jenks" else Quantiles)(values, k=k)
        bordes = np.insert(clasificador.bins, 0, values.min())
    else:
        raise ValueError(f"width_binning debe ser 'quantiles', 'fisher_jenks', 'uniform' o 'custom', no {binning!r}")
    indice = np.clip(np.searchsorted(bordes, values, side="left") - 1, 0, len(bordes) - 2)
    return indice, bordes


def _reference_values(maximum, n=3):
    """Valores redondos entre cero y `maximum` para las leyendas, con `maximum` incluido.

    Los valores redondos solos pueden quedar lejos del tope (con un máximo de
    2.340 serían 1.000 y 2.000), y entonces la muestra más gruesa de la leyenda
    no corresponde a nada del mapa. Si el último valor redondo queda bajo el
    90% del máximo, el máximo se agrega como última muestra.
    """
    from matplotlib.ticker import MaxNLocator

    valores = [float(v) for v in MaxNLocator(nbins=n).tick_values(0, maximum) if 0 < v <= maximum]
    if not valores or valores[-1] < 0.9 * maximum:
        valores.append(float(maximum))
    return valores


def _format_value(v):
    return f"{v:,.0f}" if v >= 10 else f"{v:.2g}"


def _stack_legend(ax, crear):
    """Llama a `crear`, que instala una leyenda nueva con `ax.legend`, y conserva la anterior.

    `ax.legend` reemplaza la leyenda del eje; la anterior se agrega como
    artista para que las leyendas de ancho, de tamaño de nodo y de glifos
    convivan. La última en crearse queda como leyenda del eje, que es la única
    que `savefig(bbox_inches="tight")` considera al recortar.
    """
    previa = ax.legend_
    resultado = crear()
    if previa is not None and previa not in ax.get_children():
        ax.add_artist(previa)
    return resultado


def _width_legend(ax, samples, color, hue_map, weight, legend_args):
    """Leyenda de anchos a partir de `samples` (ancho, alfa, etiqueta) y, con `hue`, un parche por categoría."""
    handles = [Line2D([], [], color=color, alpha=a, linewidth=w) for w, a, _ in samples]
    labels = [etiqueta for _, _, etiqueta in samples]
    if hue_map:
        alfa = max(a for _, a, _ in samples)
        handles = [Patch(facecolor=c, alpha=alfa) for c in hue_map.values()] + handles
        labels = list(hue_map) + labels
    opciones = {"loc": "lower left", "frameon": False, "title": weight, "handlelength": 3}
    opciones.update(legend_args or {})
    return _stack_legend(ax, lambda: ax.legend(handles, labels, **opciones))


def _palette_colorbar(ax, collection, label):
    """Barra de color, como eje inserto abajo al centro, para el color por valor."""
    from chiricoca.colors import inset_colorbar_axes

    cax = inset_colorbar_axes(ax, location="lower center", width="30%", height="3%", bbox_to_anchor=(0, 0, 1, 1), borderpad=1.2)
    barra = ax.figure.colorbar(collection, cax=cax, orientation="horizontal")
    barra.set_label(label, size="small")
    barra.ax.tick_params(labelsize="x-small")
    return barra


def _node_size_legend(ax, nodes, node_scale, node_size, node_color, node_edgecolor, weight):
    """Leyenda de tamaño de nodo: tres círculos con el total que entra y sale."""
    from chiricoca.tables.bubbles import bubble_size_legend

    valores = _reference_values(nodes["total"].max())
    tamanos = np.maximum(node_scale * np.asarray(valores), node_size)
    return _stack_legend(
        ax,
        lambda: bubble_size_legend(
            ax, tamanos, [_format_value(v) for v in valores], facecolor=node_color, edgecolor=node_edgecolor,
            location="lower right", title=f"{weight} (entra y sale)",
        ),
    )


# ---------------------------------------------------------------------------
# Glifos de nodo
# ---------------------------------------------------------------------------


def node_composition(flows, source, target, weight, by, direction="out"):
    """Tabla lugar por categoría con el peso acumulado, para `pie_glyphs`.

    Parameters
    ----------
    by : str
        Columna categórica de `flows` (el modo, el propósito, la región del
        otro extremo).
    direction : {"out", "in", "both"}
        Qué se acumula en cada lugar: lo que sale de él, lo que llega, o la
        suma de ambos.

    Returns
    -------
    DataFrame
        Indexado por lugar, una columna por categoría, en las unidades de
        `weight`. Las filas no están normalizadas; `pie_glyphs` lo hace.
    """

    def tabla(columna):
        return flows.pivot_table(
            index=columna, columns=by, values=weight, aggfunc="sum", fill_value=0, observed=True
        )

    if direction == "out":
        return tabla(source)
    if direction == "in":
        return tabla(target)
    if direction == "both":
        return tabla(source).add(tabla(target), fill_value=0)
    raise ValueError(f"direction debe ser 'out', 'in' o 'both', no {direction!r}")


def pie_glyphs(
    composition,
    palette=None,
    start_angle=90,
    edgecolor="white",
    linewidth=0.5,
    missing_color="#DDDDDD",
    legend=False,
    legend_args=None,
    zorder=3,
    n_arc=24,
):
    """Construye una función para `node_glyph` de `flow_map` que dibuja una torta en cada lugar.

    La torta ocupa el mismo círculo que ocuparía el marcador, así los flujos
    siguen llegando a su borde. Los sectores se construyen en píxeles y se
    devuelven a coordenadas de datos, con lo que quedan circulares aunque el
    CRS sea geográfico.

    Parameters
    ----------
    composition : DataFrame
        Indexado por lugar, una columna por categoría, con conteos o
        fracciones; cada fila se normaliza. `node_composition` la arma a partir
        de los flujos. Un lugar sin fila, o con todo en cero, se dibuja como un
        círculo de `missing_color`.
    palette : str, list o dict, optional
        Como en `categorical_colors`. Para que los sectores coincidan con el
        `hue` de `flow_map`, entregar la misma paleta o el mismo diccionario
        categoría a color.
    start_angle : float
        Ángulo donde empieza la primera categoría, en grados desde el este y
        contra el reloj (90 es arriba). Los sectores avanzan en sentido horario,
        en el orden de las columnas.
    legend : bool
        Agrega una leyenda categórica con `categorical_color_legend`; si el eje
        ya tiene una leyenda, la conserva. `legend_args` va a esa función.
    zorder : int
        Los flujos de `flow_map` van en 2 por omisión.

    Returns
    -------
    callable
        Función `(ax, nodes)` que dibuja las tortas y devuelve la colección.
    """
    from matplotlib.colors import to_rgba

    categorias = list(composition.columns)
    mapa, colores = categorical_colors(categorias, palette)
    rgba = np.array([to_rgba(c) for c in colores])

    def dibujar(ax, nodes):
        to_screen, to_data = _screen_transforms(ax)
        centros = to_screen(nodes[["x", "y"]].to_numpy())
        radios = _points_to_pixels(ax, nodes["radius"].to_numpy())

        partes = np.nan_to_num(composition.reindex(nodes.index).to_numpy(dtype=float), nan=0.0)
        totales = partes.sum(axis=1)
        faltan = totales <= 0
        fracciones = partes / np.where(faltan, 1, totales)[:, None]
        # un lugar sin composición es un solo sector completo de color neutro
        fracciones = np.concatenate([fracciones, faltan[:, None].astype(float)], axis=1)
        colores_sector = np.vstack([rgba, to_rgba(missing_color)])

        fin = np.cumsum(fracciones, axis=1)
        inicio = fin - fracciones
        # los sectores avanzan en sentido horario desde `start_angle`
        a0 = np.radians(start_angle) - 2 * np.pi * inicio
        a1 = np.radians(start_angle) - 2 * np.pi * fin
        t = np.linspace(0, 1, n_arc)
        angulo = a0[..., None] + (a1 - a0)[..., None] * t
        arco = centros[:, None, None, :] + radios[:, None, None, None] * np.stack(
            [np.cos(angulo), np.sin(angulo)], axis=-1
        )
        n, c = fracciones.shape
        # el primer vértice es el centro; en un sector completo se reemplaza por
        # el primer punto del arco, para que el borde no marque una costura
        centro = np.broadcast_to(centros[:, None, None, :], (n, c, 1, 2))
        completo = (fracciones >= 1 - 1e-9)[..., None, None]
        poligonos = np.concatenate([np.where(completo, arco[:, :, :1], centro), arco], axis=2)

        visibles = fracciones > 1e-9
        vertices = poligonos[visibles]
        colores_visibles = np.broadcast_to(colores_sector[None], (n, c, 4))[visibles]
        coleccion = PolyCollection(
            to_data(vertices.reshape(-1, 2)).reshape(vertices.shape),
            facecolors=colores_visibles, edgecolors=edgecolor, linewidths=linewidth, zorder=zorder,
        )
        ax.add_collection(coleccion)
        if legend:
            categorical_color_legend(ax, mapa, **{"loc": "upper left", **(legend_args or {})})
        return coleccion

    return dibujar


# ---------------------------------------------------------------------------
# Mapas
# ---------------------------------------------------------------------------


def flow_map(
    flows,
    geodf,
    source,
    target,
    weight,
    node_column=None,
    aggregate=None,
    min_weight=None,
    top_k=None,
    top_k_in=None,
    local_share=None,
    share=None,
    disparity=None,
    surplus=None,
    spanning_tree=False,
    self_loops=False,
    curvature=0.2,
    bundle=False,
    bundle_args=None,
    lanes=True,
    taper=0.0,
    arrows=False,
    arrow_scale=1.3,
    avoid_nodes=False,
    min_width=0.3,
    max_width=4.0,
    max_value=None,
    width_scale="linear",
    width_binning=None,
    width_k=5,
    width_bins=None,
    color="#0A0E50",
    palette=None,
    color_by=None,
    color_range=None,
    hue=None,
    hue_palette=None,
    hue_order=None,
    alpha=0.7,
    alpha_by=None,
    add_nodes=True,
    node_size=6,
    node_scale=None,
    node_color="white",
    node_edgecolor="#0A0E50",
    node_glyph=None,
    legend=False,
    legend_args=None,
    return_flows=False,
    ax=None,
    fig_args=None,
    **kwargs,
):
    """Dibuja flujos entre lugares como líneas de ancho proporcional al peso.

    Cada fila de `flows` es un par origen-destino con un peso. La posición de
    cada lugar sale del centroide de su geometría en `geodf`, y el ancho de la
    línea codifica el peso, proporcional desde cero: un flujo del doble de peso
    mide el doble de ancho.

    Con una matriz completa el mapa es una maraña. Para obtener un panorama de
    la red hay tres herramientas, que se combinan: reducir la cantidad de
    flujos (`min_weight`, `top_k`, `top_k_in`, `local_share`, `share`,
    `disparity`, `surplus`, `spanning_tree`, ver `filter_flows`),
    fundir la ida y la vuelta de cada par (`aggregate`, ver `aggregate_flows`) y
    agrupar los flujos que van hacia el mismo lado en haces (`bundle`, ver
    `bundle_flows`). La receta típica es `disparity=0.05` o `top_k=3` más
    `bundle=True`; con `share` se sabe qué fracción del total quedó en el mapa.

    Parameters
    ----------
    flows : DataFrame
        Una fila por par, con las columnas `source`, `target` y `weight`.
    geodf : GeoDataFrame
        Geometría de los lugares. Se usa su centroide como posición.
    source, target, weight : str
        Nombres de las columnas de `flows`.
    node_column : str, optional
        Columna de `geodf` que identifica cada lugar, una fila por lugar. Sin
        ella se usa el índice.
    aggregate : {None, "total", "net"}, optional
        Sin agregación se dibuja cada sentido por separado. Con `"total"` un
        solo trazo por par lleva la suma de ambos sentidos, sin dirección. Con
        `"net"` lleva el saldo, en el sentido dominante.
    min_weight, top_k, top_k_in, local_share, share, disparity, surplus, spanning_tree :
        Criterios de `filter_flows`, aplicados después de la agregación.
    self_loops : bool
        Los flujos de un lugar consigo mismo no se pueden dibujar. Con True
        cuentan igual en el tamaño de los nodos (`node_scale`); con False se
        descartan del todo.
    curvature : float
        Curvatura de las líneas, como fracción de su largo. Con 0 quedan
        rectas. Sin media flecha, la ida y la vuelta de un par se curvan hacia
        lados opuestos y forman una lente; con `arrows="half"` comparten la
        misma curva, una a cada lado. Se ignora con `bundle`, que traza su
        propio camino.
    bundle : bool
        Agrupa en haces los flujos que comparten corredor, con `bundle_flows`.
        Los pesos de los flujos van como `weights`, así los flujos chicos se
        pegan a los corredores de los grandes. `bundle_args` va a esa función y
        puede cambiar eso (`{"weights": None}`) o el ancho del kernel
        (`{"bandwidth": 0.06}`), que es lo que decide cuánto se agrupa.
    lanes : bool
        Desplaza cada flujo hacia su derecha la mitad de su ancho, así la ida y
        la vuelta de un mismo corredor corren en paralelo como las pistas de
        una calle en vez de una encima de la otra. Se desactiva con
        `aggregate`, donde hay un solo trazo por par, y no hace falta con
        `arrows="half"`, que ya pone cada sentido a un lado.
    taper : float
        Cuánto se afina cada flujo hacia su destino, entre 0 y 1. Con 0 el
        ancho es constante; con 1 termina en punta. Indica la dirección sin
        punta de flecha.
    arrows : {False, True, "full", "half"}
        Cómo marcar la dirección. Con `True` o `"full"` cada flujo termina en
        una punta de flecha centrada en la línea. Con `"half"` cada flujo es
        una media flecha de Tobler (1987): una banda a la derecha de la línea
        que une los dos lugares, rematada en media punta, así la ida y la
        vuelta comparten la línea, una a cada lado, y el par se lee como una
        sola flecha doble cuyo ancho total es la suma de los dos sentidos. Con
        `curvature=0` es la forma más compacta de mostrar una matriz OD. Se
        ignora con `aggregate="total"`.
    arrow_scale : float
        Largo de la punta, como múltiplo de la media geométrica entre el ancho
        del flujo y el ancho de `max_value` (`max_width`), con un piso del 25%
        de este último. La media geométrica comprime el rango: la punta de un flujo
        grande mide lo que su ancho por `arrow_scale`, y la de uno chico queda
        entre eso y su propio ancho, así ni las grandes son exageradas ni las
        chicas invisibles.
    avoid_nodes : bool
        Curva cada flujo lo justo para no pasar sobre los nodos con los que no
        conecta, porque un flujo que cruza un lugar parece detenerse en él. El
        lado y la magnitud salen de `_avoiding_curvature`; `curvature` actúa
        como mínimo. Requiere `add_nodes` y se ignora con `bundle`.
    min_width, max_width : float
        Ancho en puntos del flujo de peso cero (piso de visibilidad) y del de
        peso `max_value`.
    max_value : float, optional
        Peso que recibe `max_width`. Por omisión es el mayor de los flujos
        dibujados; fijarlo deja varios paneles en la misma escala. Los flujos
        que lo superan se dibujan más gruesos que `max_width`, en proporción,
        y la leyenda incluye al mayor de ellos.
    width_scale : {"linear", "sqrt", "log"}
        Transformación del peso antes de convertirlo en ancho. Con `"linear"`
        el ancho es proporcional; `"sqrt"` y `"log"` (logaritmo de uno más el
        peso) comprimen las colas largas, donde unos pocos flujos enormes
        dejan al resto en el piso. Se ignora con `width_binning`.
    width_binning : {None, "quantiles", "fisher_jenks", "uniform", "custom"}
        Cuantiza los anchos en `width_k` clases, con la misma convención que
        `choropleth_map`: los flujos de una clase comparten ancho, y los
        anchos de las clases van equiespaciados entre `min_width` y
        `max_width`. Con clases el ojo distingue niveles que en una escala
        continua se confunden, y la leyenda muestra el rango de peso de cada
        una. Con `"custom"`, `width_bins` trae los bordes; darlos a mano deja
        varios paneles en la misma escala.
    color : str
        Color de las líneas cuando no hay `palette` ni `hue`.
    palette : str, optional
        Paleta de seaborn con que colorear las líneas según `color_by` o, sin
        esa columna, según el peso, en la misma escala desde cero que el ancho.
    color_by : str, optional
        Columna numérica de `flows` que decide el color con `palette`, para
        mostrar una segunda variable por arista (un índice, un saldo, una
        información mutua puntual). Si toma valores de ambos signos, el rango
        de color queda simétrico en torno a cero, así una paleta divergente se
        centra donde corresponde.
    color_range : tuple of float, optional
        Límites `(vmin, vmax)` del color, para fijarlos entre paneles.
    hue : str, optional
        Columna de `flows` con una categoría por flujo (el modo, la región de
        origen) para colorear por ella. Los colores salen de
        `categorical_colors` con `hue_palette`. No se puede combinar con
        `aggregate`, porque el par agregado no tiene categoría. Una categoría
        topológica útil es si el flujo está en un triángulo, que da
        `edge_triangles`.
    hue_order : list, optional
        Orden de las categorías de `hue`. Decide el orden de la leyenda y el
        de dibujo: la última categoría queda encima de las demás, dentro de
        cada una los flujos grandes encima de los chicos.
    alpha : float, tuple of float or dict
        Transparencia. Donde varios flujos se superponen el color se acumula,
        que es lo que hace visibles los corredores. Con una tupla
        `(minima, maxima)` cada flujo recibe una transparencia proporcional a
        su peso (o a `alpha_by`), con la misma transformación o las mismas
        clases que el ancho: los flujos chicos se desvanecen y los grandes
        quedan opacos, con lo que el fondo de flujos menores deja de competir
        con la estructura principal. Con un diccionario categoría a valor,
        cada categoría de `hue` recibe la suya, para atenuar una clase de
        fondo y destacar otra.
    alpha_by : str, optional
        Columna numérica de `flows` que decide la transparencia en vez del
        peso, escalada linealmente entre su mínimo y su máximo.
    add_nodes : bool
        Dibuja un punto en cada lugar. Los flujos se acortan para no pasar por
        debajo de él, así las puntas de flecha quedan visibles.
    node_size : float
        Tamaño del marcador, en puntos al cuadrado. Con `node_scale` actúa
        como piso.
    node_scale : float, optional
        Con un valor, el área de cada nodo pasa a ser proporcional al total que
        entra y sale de él, multiplicado por este factor.
    node_glyph : callable, optional
        Función que dibuja los nodos en lugar de los círculos. Recibe
        `(ax, nodes)`, donde `nodes` es un GeoDataFrame con una fila por lugar
        dibujado, indexado por su identificador, con la geometría del centroide
        y las columnas `x` e `y` (coordenadas de datos), `size` (área del
        marcador en puntos cuadrados, la que usaría el círculo), `radius`
        (radio en puntos; los flujos se acortan hasta ahí), `outflow`, `inflow`
        y `total`. `pie_glyphs` construye una para tortas; cualquier otra
        función que dibuje sobre `ax` sirve.
    legend : bool
        Agrega las leyendas que correspondan: anchos de referencia (tres
        valores redondos, o una muestra por clase con su rango si hay
        `width_binning`), con la transparencia de cada uno y `legend_args`
        para `ax.legend` (y un parche por categoría con `hue`); una barra de
        color con `palette`; y con `node_scale` tres círculos de referencia
        para el tamaño de los nodos.
    return_flows : bool
        Devuelve además el DataFrame con los flujos que quedaron dibujados,
        para reportar cuántos son y qué fracción del total representan.

    Returns
    -------
    ax : matplotlib.axes.Axes, o (ax, flows) con `return_flows`.
    """
    if hue is not None and aggregate is not None:
        raise ValueError("hue no se puede combinar con aggregate: el par agregado no tiene categoría.")

    positions = geodf.set_index(node_column) if node_column else geodf
    if not positions.index.is_unique:
        raise ValueError("Cada lugar debe aparecer una sola vez en geodf; dissolve antes si hay partes.")
    positions = projected_centroids(positions.geometry)

    pairs = flows[flows[source].isin(positions.index) & flows[target].isin(positions.index)]
    internos = pairs[pairs[source] == pairs[target]]
    pairs = pairs[pairs[source] != pairs[target]]

    if aggregate is not None:
        pairs = aggregate_flows(pairs, source, target, weight, how=aggregate)
    pairs = filter_flows(
        pairs, source, target, weight,
        min_weight=min_weight, top_k=top_k, top_k_in=top_k_in, local_share=local_share,
        share=share, disparity=disparity, surplus=surplus, spanning_tree=spanning_tree,
    )

    if ax is None:
        fig, ax = figure_from_geodataframe(geodf, **(fig_args or {}))

    if pairs.empty:
        return (ax, pairs) if return_flows else ax

    # los flujos grandes se dibujan al final, para que queden encima; con
    # hue_order, además, cada categoría encima de las anteriores
    pairs = pairs.sort_values(weight)
    if hue is not None and hue_order is not None:
        pairs[hue] = pd.Categorical(pairs[hue], categories=hue_order)
        if pairs[hue].isna().any():
            raise ValueError("hue_order no incluye todas las categorías de hue")
        pairs = pairs.iloc[np.argsort(pairs[hue].cat.codes.to_numpy(), kind="stable")]
    pairs = pairs.reset_index(drop=True)
    values = pairs[weight].to_numpy(dtype=float)
    reference = max_value if max_value is not None else values.max()
    if reference <= 0:
        raise ValueError("Los pesos deben ser positivos.")

    transformaciones = {"linear": lambda x: np.asarray(x, dtype=float), "sqrt": np.sqrt, "log": np.log1p}
    if width_scale not in transformaciones:
        raise ValueError(f"width_scale debe ser 'linear', 'sqrt' o 'log', no {width_scale!r}")
    transformar = transformaciones[width_scale]

    def fraccion_de(v, maximo):
        return transformar(np.asarray(v, dtype=float)) / transformar(maximo)

    # la fracción del ancho máximo de cada flujo; con clases, la de su clase
    if width_binning is None:
        fraccion = fraccion_de(values, reference)
        referencias = _reference_values(reference)
        # con max_value por debajo del mayor flujo dibujado, ese flujo es más
        # grueso que max_width y la leyenda también lo muestra
        if values.max() > 1.05 * reference:
            referencias.append(float(values.max()))
        muestras = [(_format_value(v), fraccion_de(v, reference)) for v in referencias]
    else:
        clases, bordes = _classify(values, width_binning, width_k, width_bins)
        k = len(bordes) - 1
        fraccion = (clases + 1) / k
        muestras = [(f"{_format_value(bordes[i])} a {_format_value(bordes[i + 1])}", (i + 1) / k) for i in range(k)]

    def ancho_en_puntos(fraccion):
        return min_width + (max_width - min_width) * np.asarray(fraccion)

    widths = ancho_en_puntos(fraccion)
    widths_px = _points_to_pixels(ax, widths)

    if isinstance(alpha, dict):
        if hue is None:
            raise ValueError("alpha por categoría necesita hue")
        alphas = pairs[hue].map(alpha).to_numpy(dtype=float)
        if np.isnan(alphas).any():
            raise ValueError("alpha no trae valor para todas las categorías de hue")

        def alfa_de(fraccion):
            return max(alpha.values())
    elif np.ndim(alpha) == 0:
        alphas = float(alpha)

        def alfa_de(fraccion):
            return float(alpha)
    else:
        alfa_minima, alfa_maxima = alpha

        def alfa_de(fraccion):
            return alfa_minima + (alfa_maxima - alfa_minima) * np.asarray(fraccion)

        if alpha_by is None:
            alphas = alfa_de(fraccion)
        else:
            por = pairs[alpha_by].to_numpy(dtype=float)
            alphas = alfa_de((por - por.min()) / max(por.max() - por.min(), 1e-12))

    if color_by is not None:
        color_values = pairs[color_by].to_numpy(dtype=float)
        if color_range is not None:
            clim = tuple(color_range)
        elif color_values.min() < 0 < color_values.max():
            extremo = max(-color_values.min(), color_values.max())
            clim = (-extremo, extremo)
        else:
            clim = (color_values.min(), color_values.max())
    else:
        color_values = values
        clim = tuple(color_range) if color_range is not None else (0, reference)

    arrows = {True: "full", False: None}.get(arrows, arrows)
    if arrows not in (None, "full", "half"):
        raise ValueError(f"arrows debe ser False, True, 'full' o 'half', no {arrows!r}")
    if aggregate == "total":
        arrows = None

    to_screen, to_data = _screen_transforms(ax)
    coords = to_screen(np.column_stack([positions.x.to_numpy(), positions.y.to_numpy()]))
    xy = pd.DataFrame(coords, index=positions.index, columns=["x", "y"])
    origen = xy.reindex(pairs[source]).to_numpy()
    destino = xy.reindex(pairs[target]).to_numpy()

    # el tamaño de los nodos se decide antes de trazar, para acortar los flujos
    # en su borde y para esquivarlos
    usados = pd.Index(sorted(set(pairs[source]) | set(pairs[target])))
    con_internos = pd.concat([pairs, internos]) if self_loops else pairs
    salidas = con_internos.groupby(source, observed=True)[weight].sum().reindex(usados).fillna(0).to_numpy(dtype=float)
    llegadas = con_internos.groupby(target, observed=True)[weight].sum().reindex(usados).fillna(0).to_numpy(dtype=float)
    node_sizes = np.full(len(usados), float(node_size))
    if node_scale is not None:
        node_sizes = np.maximum(node_scale * (salidas + llegadas), node_size)
    puntos_nodo = positions.reindex(usados)
    nodes = gpd.GeoDataFrame(
        {
            "x": puntos_nodo.x.to_numpy(), "y": puntos_nodo.y.to_numpy(),
            "size": node_sizes, "radius": np.sqrt(node_sizes / np.pi),
            "outflow": salidas, "inflow": llegadas, "total": salidas + llegadas,
        },
        geometry=puntos_nodo.to_numpy(), index=usados, crs=positions.crs,
    )
    radios_px = _points_to_pixels(ax, nodes["radius"].to_numpy()) + 1.5
    posicion = pd.Series(np.arange(len(usados)), index=usados)
    indice_origen = posicion.reindex(pairs[source]).to_numpy()
    indice_destino = posicion.reindex(pairs[target]).to_numpy()

    if bundle:
        paths = np.asarray(bundle_flows(np.stack([origen, destino], axis=1), **{"weights": values, **(bundle_args or {})}))
    else:
        # con media flecha los dos sentidos comparten la curva, calculada en un
        # orden canónico del par y recorrida al revés para el sentido inverso
        directo = np.ones(len(pairs), dtype=bool)
        if arrows == "half":
            directo = pairs[source].to_numpy() <= pairs[target].to_numpy()
        a = np.where(directo[:, None], origen, destino)
        b = np.where(directo[:, None], destino, origen)
        curvaturas = np.full(len(pairs), float(curvature))
        if avoid_nodes and add_nodes:
            curvaturas = _avoiding_curvatures(
                a, b, xy.reindex(usados).to_numpy(), radios_px[None, :] + widths_px[:, None] / 2 + 2,
                np.column_stack([indice_origen, indice_destino]), curvature,
            )
        paths = _bezier_curves(a, b, curvaturas)
        paths = np.where(directo[:, None, None], paths, paths[:, ::-1])

    if lanes and aggregate is None and arrows != "half":
        paths = _offset_right(paths, widths_px / 2 + 0.5)

    if add_nodes:
        paths = _shorten(paths, radios_px[indice_origen], radios_px[indice_destino])

    # la punta dice la dirección, no la magnitud, que ya la lleva el ancho. Sin
    # un piso la de un flujo chico queda invisible al lado de la de uno grande
    # la referencia es el ancho de `max_value`, no el del mayor flujo dibujado:
    # un par extremo por encima de max_value no debe agrandar todas las puntas
    mayor = float(_points_to_pixels(ax, ancho_en_puntos(1.0)))
    largo_punta = arrow_scale * np.sqrt(np.maximum(widths_px, 0.25 * mayor) * mayor)
    cabezas = None
    if arrows == "full":
        cabezas, paths = _arrowheads(paths, largo_punta, np.maximum(0.43 * largo_punta, 0.8 * widths_px))

    hue_map, hue_colors = (None, None)
    if hue is not None:
        hue_map, hue_colors = categorical_colors(pairs[hue], hue_palette)

    def en_datos(vertices):
        return to_data(vertices.reshape(-1, 2)).reshape(vertices.shape)

    zorder = kwargs.pop("zorder", 2)
    if arrows == "half":
        alas = np.maximum(0.55 * largo_punta, 1.3 * widths_px)
        collection = PolyCollection(
            en_datos(_half_arrows(paths, widths_px, taper, largo_punta, alas)), zorder=zorder, **kwargs
        )
    elif taper > 0:
        collection = PolyCollection(en_datos(_tapered_polygons(paths, widths_px, taper)), zorder=zorder, **kwargs)
    else:
        collection = LineCollection(
            en_datos(paths),
            linewidths=widths,
            capstyle="butt" if (arrows or add_nodes) else "round",
            zorder=zorder,
            **kwargs,
        )
    _paint(collection, color_values, color, palette, hue_colors, clim, alphas)
    ax.add_collection(collection)

    if cabezas is not None:
        puntas = PolyCollection(en_datos(cabezas), zorder=zorder)
        _paint(puntas, color_values, color, palette, hue_colors, clim, alphas)
        ax.add_collection(puntas)

    if add_nodes and node_glyph is not None:
        node_glyph(ax, nodes)
    elif add_nodes:
        nodes.plot(
            ax=ax, color=node_color, edgecolor=node_edgecolor, markersize=nodes["size"],
            linewidth=0.5, zorder=zorder + 1,
        )

    if legend:
        etiquetas = [(ancho_en_puntos(f), alfa_de(f), texto) for texto, f in muestras]
        # cuando el color codifica otra cosa, las muestras de ancho van en gris
        neutro = color if palette is None and hue is None else "#666666"
        _width_legend(ax, etiquetas, neutro, hue_map, weight, legend_args)
        if palette is not None and hue is None:
            _palette_colorbar(ax, collection, color_by if color_by is not None else weight)
        if node_scale is not None and add_nodes:
            _node_size_legend(ax, nodes, node_scale, node_size, node_color, node_edgecolor, weight)

    return (ax, pairs) if return_flows else ax


def flow_tree(
    flows,
    geodf,
    source,
    target,
    weight,
    origin,
    node_column=None,
    direction="out",
    method="ward",
    branch_ratio=0.5,
    split_angle=100,
    smoothness=0.4,
    min_width=0.4,
    max_width=6.0,
    max_value=None,
    color="#0A0E50",
    palette=None,
    alpha=0.85,
    add_nodes=True,
    node_size=8,
    node_color="white",
    node_edgecolor="#0A0E50",
    ax=None,
    fig_args=None,
    **kwargs,
):
    """Dibuja los flujos de un lugar como un árbol que se ramifica.

    Un flow map de líneas desde un mismo origen produce un abanico donde las
    líneas se cruzan y se tapan. Acá los destinos se agrupan por cercanía y el
    flujo viaja por un tronco que se va dividiendo, así el ancho de cada rama es
    la suma de lo que va a sus hojas y hay muchos menos cruces. Es la idea de
    Phan y otros (2005), sin el ruteo que evita cruces entre ramas.

    Las ramas son curvas de Bézier cúbicas con la misma tangente en cada
    bifurcación: la rama que llega y las que salen comparten dirección, y las
    que salen parten una al lado de la otra repartiéndose el ancho de la que
    llega. Con eso el flujo se ve continuo, como un río que se abre en brazos,
    y no como una ruta con quiebres, que sugeriría una estructura que los datos
    no tienen.

    Parameters
    ----------
    origin : hashable
        El lugar raíz del árbol. Con `direction="out"` es donde nacen los
        flujos y solo se usan las filas cuyo `source` es este; con
        `direction="in"` es donde llegan y se usan las filas cuyo `target` es
        este, con lo que el árbol muestra de dónde viene lo que recibe.
    method : str
        Método de agrupamiento jerárquico que arma el árbol, de
        `scipy.cluster.hierarchy.linkage`.
    branch_ratio : float
        Qué fracción del camino hacia su grupo avanza cada bifurcación, entre 0
        y 1. Con valores bajos el árbol se ramifica tarde y las ramas quedan
        largas; con valores altos se abre temprano y se parece a un abanico.
    split_angle : float
        Ángulo, en grados, entre las direcciones en que salen de la raíz dos
        grupos de hojas a partir del cual cada grupo recibe su propio tronco.
        Con una raíz central, rodeada de hojas, un tronco único tendría que
        devolverse; con 180 siempre hay un tronco único.
    smoothness : float
        Distancia de los puntos de control de cada rama, como fracción de su
        largo. Con valores bajos las ramas van casi rectas entre bifurcaciones
        y el quiebre reaparece; con valores altos se abren en curvas amplias.
    min_width, max_width : float
        Ancho en puntos de la rama sin flujo y de la que lleva `max_value`. El
        ancho es proporcional al valor desde cero, y eso es lo que hace que el
        tronco se vea como la suma de sus ramas.
    max_value : float, optional
        Flujo que recibe `max_width`. Por omisión es el total del árbol, o sea
        el ancho del tronco. Fijarlo en el mismo valor deja varios árboles
        comparables entre sí.
    palette : str, optional
        Paleta con que colorear cada rama según su flujo. Sin ella el árbol se
        dibuja como una sola silueta del color `color`.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    import shapely
    from scipy.cluster.hierarchy import linkage
    from shapely.geometry import LineString, Point
    from shapely.ops import unary_union

    if direction not in ("out", "in"):
        raise ValueError(f"direction debe ser 'out' o 'in', no {direction!r}")
    raiz_col, hoja_col = (source, target) if direction == "out" else (target, source)

    positions = geodf.set_index(node_column) if node_column else geodf
    if not positions.index.is_unique:
        raise ValueError("Cada lugar debe aparecer una sola vez en geodf; dissolve antes si hay partes.")
    positions = projected_centroids(positions.geometry)

    ramas = flows[(flows[raiz_col] == origin) & (flows[hoja_col] != origin)]
    ramas = ramas[ramas[hoja_col].isin(positions.index)]
    if origin not in positions.index or len(ramas) < 2:
        raise ValueError("Hacen falta al menos dos lugares conocidos para armar el árbol.")

    if ax is None:
        fig, ax = figure_from_geodataframe(geodf, **(fig_args or {}))
    to_screen, to_data = _screen_transforms(ax)

    hojas = list(ramas[hoja_col])
    coordenadas = to_screen(np.column_stack([positions[hojas].x, positions[hojas].y]))
    pesos = ramas[weight].to_numpy(dtype=float)

    # El agrupamiento jerárquico da la forma del árbol: qué hojas comparten
    # rama y en qué orden se separan. Los nodos 0..n-1 son las hojas y los
    # siguientes las bifurcaciones, en el orden en que las crea linkage.
    n = len(hojas)
    cargas = list(pesos)
    centros = list(coordenadas)
    hijos = {}
    for paso, (a, b, _, _) in enumerate(linkage(coordenadas, method=method)):
        a, b = int(a), int(b)
        peso = cargas[a] + cargas[b]
        cargas.append(peso)
        centros.append((centros[a] * cargas[a] + centros[b] * cargas[b]) / peso)
        hijos[n + paso] = (a, b)

    # La posición de una bifurcación no es el centroide de su grupo: puesta
    # ahí, el tronco parte hacia el centro de la nube y las ramas tienen que
    # devolverse, con lo que el árbol se cruza consigo mismo. Cada bifurcación
    # se ubica recorriendo desde la raíz hacia afuera, a una fracción del
    # camino entre su padre y el centro de su grupo, así el flujo avanza en una
    # sola dirección y se va abriendo.
    raiz_xy = to_screen([[positions[origin].x, positions[origin].y]])[0]
    puntos = list(centros)

    def unitario(v):
        norma = np.hypot(*v)
        return v / norma if norma > 0 else v

    # Un tronco por cada grupo de hojas que sale de la raíz en una dirección
    # distinta. Con un tronco único, una raíz rodeada de hojas tendría que
    # mandar el flujo hacia un lado y devolverlo.
    troncos = [len(puntos) - 1]
    limite = np.cos(np.radians(split_angle))
    cambio = True
    while cambio:
        cambio = False
        for k in list(troncos):
            if k >= n:
                a, b = hijos[k]
                if unitario(centros[a] - raiz_xy) @ unitario(centros[b] - raiz_xy) < limite:
                    troncos.remove(k)
                    troncos.extend([a, b])
                    del hijos[k]
                    cambio = True

    pendientes = []
    for k in troncos:
        if k >= n:
            puntos[k] = raiz_xy + branch_ratio * (centros[k] - raiz_xy)
            pendientes.append(k)
    while pendientes:
        k = pendientes.pop()
        for h in hijos[k]:
            if h >= n:
                puntos[h] = puntos[k] + branch_ratio * (centros[h] - puntos[k])
                pendientes.append(h)

    # el lugar raíz entra al árbol como un nodo más, padre de los troncos
    raiz = len(puntos)
    puntos.append(raiz_xy)
    cargas.append(sum(cargas[k] for k in troncos))
    hijos[raiz] = tuple(troncos)
    padre = {h: k for k, ramas_k in hijos.items() for h in ramas_k}

    # la tangente de cada nodo promedia la dirección con que llega el flujo y la
    # dirección media, ponderada por carga, con que sale
    tangentes = {}
    for k in set(padre) | {raiz}:
        partes = []
        if k in padre:
            partes.append(unitario(puntos[k] - puntos[padre[k]]))
        if k in hijos:
            partes.append(unitario(sum(unitario(puntos[h] - puntos[k]) * cargas[h] for h in hijos[k])))
        suma = sum(partes)
        tangentes[k] = unitario(suma) if np.hypot(*suma) > 1e-9 else partes[-1]

    reference = max_value if max_value is not None else cargas[raiz]

    def ancho(k):
        return _points_to_pixels(ax, min_width + (max_width - min_width) * cargas[k] / reference)

    t = np.linspace(0, 1, 64)[:, None]
    formas, valores = [], []

    def rama(p0, tangente_inicio, h):
        w = ancho(h)
        p3 = puntos[h]
        largo = np.hypot(*(p3 - p0))
        p1 = p0 + tangente_inicio * smoothness * largo
        p2 = p3 - tangentes[h] * smoothness * largo
        curva = (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3
        forma = LineString(curva).buffer(w / 2, cap_style="flat")
        if h < n:
            # la hoja termina redondeada, no con un corte recto
            forma = forma.union(Point(p3).buffer(w / 2))
        formas.append(forma)
        valores.append(cargas[h])

    for k, ramas_k in hijos.items():
        if k == raiz:
            # los troncos salen del centro de la raíz, cada uno en su dirección
            for h in ramas_k:
                rama(raiz_xy, unitario(puntos[h] - raiz_xy), h)
            continue

        tangente = tangentes[k]
        izquierda = np.array([-tangente[1], tangente[0]])
        # las ramas salen ordenadas de izquierda a derecha según hacia dónde van,
        # cada una centrada en su franja del ancho que llega
        orden = sorted(
            ramas_k,
            key=lambda h: -(tangente[0] * (puntos[h] - puntos[k])[1] - tangente[1] * (puntos[h] - puntos[k])[0]),
        )
        desplazamiento = sum(ancho(h) for h in orden) / 2
        hacia_atras = False
        for h in orden:
            w = ancho(h)
            p0 = puntos[k] + izquierda * (desplazamiento - w / 2)
            desplazamiento -= w
            direccion = unitario(puntos[h] - p0)
            coseno = tangente @ direccion
            tangente_inicio = tangente
            if coseno < 0:
                # una rama que sale hacia atrás daría un bucle si partiera con la
                # tangente del tronco: su tangente se inclina hacia su destino
                tangente_inicio = unitario(tangente * (1 + coseno) - direccion * coseno)
                hacia_atras = True
            rama(p0, tangente_inicio, h)
        if hacia_atras:
            # un disco tapa la muesca que deja la rama que no comparte tangente
            formas.append(Point(puntos[k]).buffer(ancho(k) / 2))
            valores.append(cargas[k])

    zorder = kwargs.pop("zorder", 2)
    if palette is not None:
        # una rama puede quedar en varias piezas si su curva se cruza a sí misma
        piezas = [
            (parte, valores[i])
            for i in np.argsort(valores)
            for parte in shapely.get_parts(formas[i])
        ]
        coleccion = PolyCollection(
            [to_data(np.asarray(parte.exterior.coords)) for parte, _ in piezas],
            alpha=alpha, zorder=zorder, **kwargs,
        )
        coleccion.set_cmap(colormap_from_palette(palette, n_colors=256))
        coleccion.set_array(np.array([v for _, v in piezas]))
        coleccion.set_edgecolor("none")
        ax.add_collection(coleccion)
    else:
        # una sola silueta, así la transparencia no marca las uniones
        silueta = shapely.transform(unary_union([f.buffer(0.05) for f in formas]), to_data)
        gpd.GeoSeries([silueta], crs=geodf.crs).plot(
            ax=ax, facecolor=color, edgecolor="none", alpha=alpha, zorder=zorder, **kwargs
        )

    if add_nodes:
        lugares = gpd.GeoDataFrame(geometry=positions[[origin] + hojas].to_numpy(), crs=positions.crs)
        tamanos = [node_size * 3] + [node_size] * len(hojas)
        lugares.plot(ax=ax, color=node_color, edgecolor=node_edgecolor,
                     markersize=tamanos, linewidth=0.5, zorder=zorder + 1)

    return ax


def line_heat_map(
    geodf,
    weight=None,
    segment_length=60,
    snap=30,
    min_width=0.2,
    max_width=4.5,
    palette="magma_r",
    gamma=0.5,
    return_collection=False,
    ax=None,
    fig_args=None,
    **kwargs,
):
    """Acumula el peso de muchos trazados sobre los segmentos que comparten.

    Dibujar cada trazado por separado tapa los que van debajo, así que un eje
    por donde pasan cuarenta recorridos se ve igual que una calle por donde pasa
    uno. Esto suma en vectorial antes de dibujar: cada trazado se densifica en
    segmentos cortos, sus vértices se redondean a una grilla, y los segmentos
    que coinciden acumulan el peso de todos los trazados que los recorren.

    Parameters
    ----------
    geodf : GeoDataFrame
        Trazados, como LineString o MultiLineString, en un CRS métrico.
    weight : str, optional
        Columna con el peso de cada trazado. Sin ella cada uno pesa 1 y el
        resultado cuenta cuántos pasan por cada segmento.
    segment_length : float
        Largo máximo de los segmentos en que se parte cada trazado, en las
        unidades del CRS.
    snap : float
        Tamaño de la grilla a la que se redondean los vértices. Es lo que hace
        que dos trazados paralelos digitalizados con distinto muestreo caigan
        sobre el mismo segmento en vez de quedar al lado.
    gamma : float
        Exponente de la escala de color y de grosor. Con 0,5, que es la raíz
        cuadrada, la mediana cae cerca del 10% de la rampa y solo los ejes
        troncales se oscurecen; con escala logarítmica casi toda la red rinde
        el mismo color.

    Returns
    -------
    ax, o (ax, collection) si `return_collection`, para ponerle una barra de
    color.
    """
    import shapely
    from matplotlib.colors import PowerNorm

    if ax is None:
        fig, ax = figure_from_geodataframe(geodf, **(fig_args or {}))

    pesos_entrada = geodf[weight].to_numpy(dtype=float) if weight else np.ones(len(geodf))

    # una fila por LineString, con el peso de su trazado
    partes = geodf.geometry.reset_index(drop=True).explode(index_parts=False)
    pesos_parte = pesos_entrada[partes.index.to_numpy()]
    validas = (partes.geom_type == "LineString").to_numpy() & (partes.length > 0).to_numpy() & (pesos_parte > 0)
    partes, pesos_parte = partes[validas], pesos_parte[validas]

    if partes.empty:
        return (ax, None) if return_collection else ax

    densas = shapely.segmentize(partes.to_numpy(), segment_length)
    coords, indice = shapely.get_coordinates(densas, return_index=True)
    nodos = np.round(coords / snap).astype(np.int64)

    # segmentos consecutivos dentro de un mismo trazado, sin los degenerados
    a, b = nodos[:-1], nodos[1:]
    mismo = (indice[:-1] == indice[1:]) & (a != b).any(axis=1)
    a, b, pesos_segmento = a[mismo], b[mismo], pesos_parte[indice[:-1][mismo]]

    # la clave del segmento no depende del sentido en que se recorre
    invertir = (a[:, 0] > b[:, 0]) | ((a[:, 0] == b[:, 0]) & (a[:, 1] > b[:, 1]))
    claves = np.where(invertir[:, None], np.hstack([b, a]), np.hstack([a, b]))
    unicas, inversa = np.unique(claves, axis=0, return_inverse=True)
    pesos = np.bincount(inversa.ravel(), weights=pesos_segmento)

    segmentos = unicas.reshape(-1, 2, 2).astype(float) * snap

    # los segmentos más cargados se dibujan al final, para que queden encima
    orden = np.argsort(pesos)
    segmentos, pesos = segmentos[orden], pesos[orden]

    fraccion = (pesos / pesos.max()) ** gamma
    coleccion = LineCollection(
        segmentos,
        cmap=colormap_from_palette(palette, n_colors=256),
        norm=PowerNorm(gamma=gamma, vmin=0, vmax=pesos.max()),
        capstyle="round",
        linewidths=min_width + (max_width - min_width) * fraccion,
        **kwargs,
    )
    coleccion.set_array(pesos)
    ax.add_collection(coleccion)

    return (ax, coleccion) if return_collection else ax
