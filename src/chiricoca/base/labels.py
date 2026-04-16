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


def layout_labels_vertical(ax, y_positions, labels, fontsize=8, padding=0.005,
                           iterations=50, y_min=None, y_max=None):
    """
    Calcula posiciones Y para etiquetas evitando solapamiento.

    Desplaza etiquetas verticalmente cuando es necesario, minimizando
    el desplazamiento total respecto a las posiciones originales.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes donde se medirán las etiquetas.
    y_positions : array-like
        Posiciones Y originales de cada etiqueta.
    labels : list of str
        Textos de las etiquetas.
    fontsize : int
        Tamaño de fuente para medir la altura.
    padding : float
        Espacio mínimo entre etiquetas en coordenadas de datos.
    iterations : int
        Número de órdenes aleatorios a probar.
    y_min : float, optional
        Límite inferior del área de etiquetas. Si es None, usa el límite
        inferior del eje.
    y_max : float, optional
        Límite superior del área de etiquetas. Si es None, usa el límite
        superior del eje.

    Returns
    -------
    list of float
        Posiciones Y ajustadas para cada etiqueta.
    """
    fig = ax.get_figure()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    if y_min is None or y_max is None:
        ax_ymin, ax_ymax = ax.get_ylim()
        if y_min is None:
            y_min = ax_ymin
        if y_max is None:
            y_max = ax_ymax

    y_range = y_max - y_min
    step = y_range * 0.005

    heights = []
    for y, label in zip(y_positions, labels):
        t = ax.text(0, y, label, va='center', fontsize=fontsize)
        bbox = t.get_window_extent(renderer=renderer).transformed(ax.transData.inverted())
        heights.append(bbox.y1 - bbox.y0)
        t.remove()

    n = len(labels)
    y_list = list(y_positions)

    def solve_greedy(order):
        y_finals = [None] * n
        occupied = []

        for i in order:
            y_orig = y_list[i]
            h = heights[i]

            best_y = y_orig
            for offset in [0] + [d * s for d in range(1, 100) for s in [step, -step]]:
                y_try = y_orig + offset
                if y_try - h/2 < y_min or y_try + h/2 > y_max:
                    continue
                b, t = y_try - h/2 - padding, y_try + h/2 + padding
                if not any(ob < t and ot > b for ob, ot in occupied):
                    best_y = y_try
                    break

            y_finals[i] = best_y
            occupied.append((best_y - h/2 - padding, best_y + h/2 + padding))

        return y_finals

    def total_displacement(y_finals):
        return sum(abs(yf - yo) for yf, yo in zip(y_finals, y_list))

    def dist_to_boundary(i):
        return min(y_list[i] - y_min, y_max - y_list[i])

    orders = [
        list(range(n)),
        list(range(n-1, -1, -1)),
        sorted(range(n), key=lambda i: heights[i], reverse=True),
        sorted(range(n), key=lambda i: y_list[i]),
        sorted(range(n), key=lambda i: y_list[i], reverse=True),
        sorted(range(n), key=lambda i: dist_to_boundary(i)),
    ]

    rng = np.random.default_rng(42)
    for _ in range(iterations):
        orders.append(rng.permutation(n).tolist())

    best_y_finals = None
    best_score = float('inf')

    for order in orders:
        y_finals = solve_greedy(order)
        score = total_displacement(y_finals)
        if score < best_score:
            best_score = score
            best_y_finals = y_finals

    return best_y_finals


def _draw_legend_right(ax, df, positions, widths, colors):
    """Dibuja leyenda lateral con líneas y codos."""
    y_origins = []
    bottom = 0
    for col in df.columns:
        height = df[col].iloc[-1]
        y_origins.append(bottom + height / 2)
        bottom += height
    
    x_right = positions[-1] + widths.iloc[-1]
    y_finals = layout_labels_vertical(ax, y_origins, df.columns.tolist(), fontsize=8)
    
    x_range = x_right
    tick_length = x_range * 0.05
    box_size = 0.02
    box_width = x_range * 0.02
    threshold = 0.01
    
    codo_indices = []
    codo_count = 0
    for y_orig, y_final in zip(y_origins, y_finals):
        if abs(y_final - y_orig) < threshold:
            codo_indices.append(None)
        else:
            codo_indices.append(codo_count)
            codo_count += 1
    
    for y_orig, y_final, label, color, codo_idx in zip(y_origins, y_finals, df.columns, colors, codo_indices):
        x_end = x_right + tick_length
        
        if codo_idx is None:
            ax.plot([x_right, x_end], [y_final, y_final], color='black', lw=0.5, clip_on=False)
        else:
            x_codo = x_right + tick_length * (0.2 + 0.15 * codo_idx)
            x_codo = min(x_codo, x_right + tick_length * 0.8)
            ax.plot([x_right, x_codo], [y_orig, y_orig], color='black', lw=0.5, clip_on=False)
            ax.plot([x_codo, x_codo], [y_orig, y_final], color='black', lw=0.5, clip_on=False)
            ax.plot([x_codo, x_end], [y_final, y_final], color='black', lw=0.5, clip_on=False)
        
        ax.add_patch(plt.Rectangle((x_end + box_width*0.3, y_final - box_size/2), 
                                   box_width, box_size,
                                   facecolor=color, edgecolor='black', lw=0.5, clip_on=False))
        
        ax.text(x_end + box_width*1.5, y_final, label, va='center', ha='left', fontsize=8,
                bbox=dict(facecolor='white', edgecolor='none', pad=1, alpha=0.7))
