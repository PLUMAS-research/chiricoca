import numpy as np
import matplotlib.pyplot as plt

def layout_labels_horizontal(ax, x_positions, labels, fontsize=8, padding_frac=0.01, iterations=50):
    """
    Asigna niveles verticales a etiquetas horizontales para evitar solapamiento.
    
    Usa un algoritmo greedy que prueba múltiples órdenes de inserción y selecciona
    la que minimiza cruces entre líneas de conexión.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes donde se medirán las etiquetas.
    x_positions : array-like
        Posiciones X de cada etiqueta.
    labels : list of str
        Textos de las etiquetas.
    fontsize : int
        Tamaño de fuente para medir el ancho.
    padding_frac : float
        Fracción del rango X para padding entre etiquetas.
    iterations : int
        Número de órdenes aleatorios a probar.
    
    Returns
    -------
    list of int
        Nivel asignado a cada etiqueta (0 = más cercano al eje).
    """
    fig = ax.get_figure()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    
    x_range = max(x_positions) - min(x_positions)
    padding = x_range * padding_frac
    
    intervals = []
    for x, label in zip(x_positions, labels):
        t = ax.text(x, 0, label, ha='center', fontsize=fontsize)
        bbox = t.get_window_extent(renderer=renderer).transformed(ax.transData.inverted())
        intervals.append((bbox.x0 - padding, bbox.x1 + padding))
        t.remove()
    
    n = len(labels)
    x_list = list(x_positions)
    
    def solve_greedy(order):
        levels = [None] * n
        level_occupancy = []
        
        for i in order:
            left, right = intervals[i]
            x = x_list[i]
            best_level = None
            best_cost = float('inf')
            
            for level in range(len(level_occupancy) + 1):
                if level < len(level_occupancy):
                    if any(l < right and r > left for l, r in level_occupancy[level]):
                        continue
                
                cost = sum(1 for lv in range(level) 
                          for l, r in level_occupancy[lv] if l < x < r)
                cost += level * 0.001
                
                if cost < best_cost:
                    best_cost = cost
                    best_level = level
            
            levels[i] = best_level
            while len(level_occupancy) <= best_level:
                level_occupancy.append([])
            level_occupancy[best_level].append((left, right))
        
        return levels
    
    def count_crossings(levels):
        total = 0
        for i in range(n):
            for j in range(n):
                if i != j and levels[j] < levels[i]:
                    left, right = intervals[j]
                    if left < x_list[i] < right:
                        total += 1
        return total
    
    orders = [
        list(range(n)),
        list(range(n-1, -1, -1)),
        sorted(range(n), key=lambda i: intervals[i][1] - intervals[i][0], reverse=True),
        sorted(range(n), key=lambda i: x_list[i]),
    ]
    
    rng = np.random.default_rng(42)
    for _ in range(iterations):
        orders.append(rng.permutation(n).tolist())
    
    best_levels = None
    best_crossings = float('inf')
    
    for order in orders:
        levels = solve_greedy(order)
        crossings = count_crossings(levels)
        if crossings < best_crossings:
            best_crossings = crossings
            best_levels = levels
    
    return best_levels


def measure_label_heights(ax, labels, fontsize=8, renderer=None):
    """Alto de cada rótulo en coordenadas de datos del eje y.

    Dibuja la figura una vez si no recibe `renderer`; con varias mediciones
    conviene obtenerlo una sola vez con `fig.canvas.get_renderer()`.
    """
    if renderer is None:
        fig = ax.get_figure()
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
    alturas = []
    for label in labels:
        t = ax.text(0, 0, str(label), va="center", fontsize=fontsize)
        bbox = t.get_window_extent(renderer=renderer).transformed(ax.transData.inverted())
        alturas.append(abs(bbox.y1 - bbox.y0))
        t.remove()
    return np.asarray(alturas, dtype=float)


def _isotonic(z):
    """Regresión isotónica no decreciente de `z` con pesos iguales (agrupar violadores adyacentes)."""
    bloques = []  # [media, cantidad, inicio]
    for i, valor in enumerate(z):
        bloques.append([float(valor), 1, i])
        while len(bloques) > 1 and bloques[-2][0] > bloques[-1][0]:
            media_2, n_2, _ = bloques.pop()
            media_1, n_1, inicio = bloques.pop()
            bloques.append([(media_1 * n_1 + media_2 * n_2) / (n_1 + n_2), n_1 + n_2, inicio])
    salida = np.empty(len(z))
    for media, cantidad, inicio in bloques:
        salida[inicio : inicio + cantidad] = media
    return salida


def layout_labels_vertical(ax, y_positions, labels, fontsize=8, padding=0.005,
                           iterations=None, y_min=None, y_max=None, renderer=None, heights=None):
    """Posiciones y para rótulos que no se solapen, conservando el orden de sus anclas.

    Minimiza el desplazamiento cuadrático total sujeto a que dos rótulos
    consecutivos disten al menos la suma de sus medias alturas más `padding`.
    Con el cambio de variable `z_i = y_i - c_i`, donde `c_i` acumula esas
    separaciones, la restricción se vuelve `z` no decreciente y la solución es
    la regresión isotónica, exacta y lineal en la cantidad de rótulos. Como el
    orden de las anclas se conserva, las líneas guía nunca se cruzan.

    Si el bloque de rótulos cabe entre `y_min` y `y_max` (los límites del eje
    por omisión) se corre para quedar dentro; si no cabe, se centra sobre las
    anclas y sobresale del eje.

    Parameters
    ----------
    y_positions : array-like
        Anclas, en coordenadas de datos.
    labels : list of str
        Textos, para medir su alto.
    fontsize : int
        Tamaño de fuente con que se miden.
    padding : float
        Separación mínima entre rótulos, en coordenadas de datos.
    iterations : deprecado
        Sin efecto; la solución ya no es iterativa.
    renderer : optional
        Renderer de la figura, para no redibujarla al medir.
    heights : array-like, optional
        Altos ya medidos, en coordenadas de datos; ahorra la medición.

    Returns
    -------
    list of float
        Posición y de cada rótulo, en el orden de entrada.
    """
    y = np.asarray(y_positions, dtype=float)
    n = len(y)
    if n == 0:
        return []
    if y_min is None or y_max is None:
        ax_ymin, ax_ymax = ax.get_ylim()
        y_min = ax_ymin if y_min is None else y_min
        y_max = ax_ymax if y_max is None else y_max
    alturas = np.asarray(heights, dtype=float) if heights is not None else measure_label_heights(ax, labels, fontsize, renderer)

    orden = np.argsort(y, kind="stable")
    anclas, h = y[orden], alturas[orden]
    separaciones = np.concatenate([[0.0], np.cumsum((h[:-1] + h[1:]) / 2 + padding)])
    finales = _isotonic(anclas - separaciones) + separaciones

    bajo, alto = finales[0] - h[0] / 2, finales[-1] + h[-1] / 2
    if alto - bajo <= y_max - y_min:
        finales += max(0.0, y_min - bajo) - max(0.0, alto - y_max)
    else:
        finales += anclas.mean() - finales.mean()

    salida = np.empty(n)
    salida[orden] = finales
    return salida.tolist()


def draw_right_legend(
    ax,
    y_origins,
    labels,
    colors,
    x_right,
    fontsize=8,
    line_color="0.3",
    leader="straight",
    sizes=None,
    overflow="shrink",
    min_fontsize=5,
    group_label="otras",
    padding_frac=0.005,
    renderer=None,
):
    """Leyenda lateral: cada serie se rotula a la derecha del gráfico, a la altura en que termina.

    Es la leyenda de los gráficos apilados (`barchart`, `marimekko`,
    `streamgraph`). Las posiciones salen de `layout_labels_vertical`, que
    conserva el orden de las series, así que las guías no se cruzan. Las
    distancias se miden como fracciones de los rangos de los ejes y la leyenda
    se ve igual con datos de cualquier escala.

    Parameters
    ----------
    y_origins : array-like
        Centro de cada serie en el borde derecho, en coordenadas de datos.
    labels, colors : array-like
        Rótulo y color de cada serie, en el mismo orden.
    x_right : float
        Borde derecho del gráfico, en coordenadas de datos (numéricas; una
        fecha va como número de matplotlib).
    leader : {"straight", "elbow"}
        Guía recta del ancla al rótulo, o con un codo a mitad de camino.
    sizes : array-like, optional
        Magnitud de cada serie en el borde derecho; decide cuáles conservar con
        `overflow="group"`. Sin ella se estima con la separación entre anclas.
    overflow : {"shrink", "group", "spill"}
        Qué hacer cuando los rótulos no caben en el alto del eje: `"shrink"`
        reduce la fuente hasta `min_fontsize`; `"group"` rotula solo las series
        mayores que caben y reúne las demás bajo `group_label` con su cantidad,
        marcando con una llave el tramo que ocupan si son contiguas; `"spill"`
        los deja sobresalir del eje, centrados en sus anclas.

    Returns
    -------
    list of float
        Posición y final de cada rótulo conservado, en el orden de entrada
        (`nan` para los agrupados).
    """
    if leader not in ("straight", "elbow"):
        raise ValueError(f"leader debe ser 'straight' o 'elbow', no {leader!r}")
    if overflow not in ("shrink", "group", "spill"):
        raise ValueError(f"overflow debe ser 'shrink', 'group' o 'spill', no {overflow!r}")

    if renderer is None:
        fig = ax.get_figure()
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

    y = np.asarray(y_origins, dtype=float)
    textos = [str(l) for l in labels]
    n = len(y)
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    x_range, y_range = x_max - x_min, y_max - y_min
    padding = padding_frac * y_range

    def cabe(alturas):
        return alturas.sum() + padding * (len(alturas) - 1) <= y_range

    alturas = measure_label_heights(ax, textos, fontsize, renderer)
    conservadas = np.arange(n)
    agrupadas = np.array([], dtype=int)
    if not cabe(alturas):
        if overflow == "shrink":
            while fontsize > min_fontsize and not cabe(alturas):
                fontsize = max(min_fontsize, fontsize * 0.9)
                alturas = measure_label_heights(ax, textos, fontsize, renderer)
        elif overflow == "group":
            if sizes is None:
                # sin tamaños, el espacio que rodea a cada ancla hace de tamaño
                ordenadas = np.sort(y)
                huecos = np.diff(np.concatenate([[2 * ordenadas[0] - ordenadas[1]], ordenadas, [2 * ordenadas[-1] - ordenadas[-2]]]))
                tamanos = np.empty(n)
                tamanos[np.argsort(y, kind="stable")] = (huecos[:-1] + huecos[1:]) / 2
            else:
                tamanos = np.asarray(sizes, dtype=float)
            por_tamano = np.argsort(-tamanos, kind="stable")
            alto_grupo = measure_label_heights(ax, [f"{group_label} ({n})"], fontsize, renderer)[0]
            k = n - 1
            while k > 0 and alturas[por_tamano[:k]].sum() + alto_grupo + padding * k > y_range:
                k -= 1
            conservadas = np.sort(por_tamano[:k])
            agrupadas = np.sort(por_tamano[k:])

    anclas = list(y[conservadas])
    rotulos = [textos[i] for i in conservadas]
    altos = list(alturas[conservadas])
    if len(agrupadas):
        anclas.append(float(y[agrupadas].mean()))
        rotulos.append(f"{group_label} ({len(agrupadas)})")
        altos.append(measure_label_heights(ax, rotulos[-1:], fontsize, renderer)[0])
    finales = layout_labels_vertical(
        ax, anclas, rotulos, fontsize=fontsize, padding=padding, heights=altos, renderer=renderer
    )

    tick = 0.05 * x_range
    box_width = 0.02 * x_range
    box_height = 0.02 * y_range
    x_end = x_right + tick

    def guia(y_o, y_f):
        if leader == "elbow" and abs(y_f - y_o) > 0.01 * y_range:
            x_codo = x_right + tick * 0.5
            ax.plot([x_right, x_codo, x_codo, x_end], [y_o, y_o, y_f, y_f], color=line_color, lw=0.5, clip_on=False)
        else:
            ax.plot([x_right, x_end], [y_o, y_f], color=line_color, lw=0.5, clip_on=False)

    def rotulo(y_f, texto, color):
        ax.add_patch(plt.Rectangle(
            (x_end + box_width * 0.3, y_f - box_height / 2), box_width, box_height,
            facecolor=color, edgecolor=line_color, lw=0.5, clip_on=False,
        ))
        ax.text(x_end + box_width * 1.5, y_f, texto, va="center", ha="left", fontsize=fontsize, clip_on=False)

    colores = list(colors)
    for i, y_f in zip(conservadas, finales):
        guia(y[i], y_f)
        rotulo(y_f, textos[i], colores[i])

    if len(agrupadas):
        y_f = finales[-1]
        bajo, alto = y[agrupadas].min(), y[agrupadas].max()
        contiguas = not ((y[conservadas] > bajo) & (y[conservadas] < alto)).any()
        if contiguas and len(agrupadas) > 1:
            # una llave abarca el tramo de las series agrupadas
            x_llave = x_right + tick * 0.35
            ax.plot([x_llave, x_llave], [bajo, alto], color=line_color, lw=0.5, clip_on=False)
            ax.plot([x_llave, x_end], [(bajo + alto) / 2, y_f], color=line_color, lw=0.5, clip_on=False)
        else:
            guia(y[agrupadas].mean(), y_f)
        rotulo(y_f, rotulos[-1], "0.75")

    salida = np.full(n, np.nan)
    salida[conservadas] = finales[: len(conservadas)]
    return salida.tolist()
