import numpy as np
from shapely.affinity import scale as _scale
from shapely.geometry import Point

from chiricoca.geo.figures import figure_from_geodataframe
from chiricoca.geo.utils import projected_centroids


def _projected(geodf):
    """Devuelve el GeoDataFrame en un CRS métrico y el CRS original.

    Los dos cartogramas trabajan con áreas y distancias, que en grados no
    significan nada.
    """
    original = geodf.crs
    if original is not None and original.is_geographic:
        return geodf.to_crs(geodf.estimate_utm_crs()), original
    return geodf, None


def non_contiguous_cartogram(geodf, column, max_scale=1.0):
    """Escala cada polígono en torno a su centroide hasta que su área es proporcional a `column`.

    Es el cartograma de Olson (1976): el polígono más denso (el de mayor valor
    por unidad de área) sirve de ancla y conserva su tamaño, y todos los demás
    se achican hasta que el área de cada uno es proporcional a su valor. Así el
    mapa deja de mentir sobre el peso de cada área. A cambio los polígonos se
    separan y el mapa deja de ser contiguo, que es de donde viene el nombre.

    `max_scale` es el factor que recibe el ancla: con 1 queda de su tamaño
    original; con menos, el cartograma entero se achica en la misma proporción.
    """
    proyectado, original = _projected(geodf)
    valores = proyectado[column].to_numpy(dtype=float)
    if (valores < 0).any():
        raise ValueError(f"'{column}' tiene valores negativos y no se puede escalar por área.")

    areas = proyectado.area.to_numpy()
    densidad = np.divide(valores, areas, out=np.zeros_like(valores), where=areas > 0)
    # el área objetivo es valor / densidad del ancla, y como el área crece con
    # el cuadrado del factor, el factor va en raíz
    objetivo = np.divide(valores, densidad.max() * areas, out=np.zeros_like(valores), where=areas > 0)
    factores = max_scale * np.sqrt(objetivo)

    salida = proyectado.copy()
    salida["geometry"] = [
        _scale(geom, xfact=f, yfact=f, origin="centroid")
        for geom, f in zip(proyectado.geometry, factores)
    ]
    return salida.to_crs(original) if original else salida


def dorling_cartogram(
    geodf,
    column,
    max_radius=None,
    iterations=600,
    anchor_stiffness=0.05,
    damping=0.6,
    collision_slop=1e-4,
):
    """Reemplaza cada área por un círculo de área proporcional a `column`.

    Los círculos parten en el centroide de su área y se acomodan con el motor
    de física que chiricoca ya usa para `bubble_plot`: colisionan entre sí y un
    resorte los sujeta a su posición original. Se pierde la forma de cada área
    y se gana que el tamaño sea comparable sin que un círculo tape a otro.

    Parameters
    ----------
    max_radius : float, optional
        Radio del círculo mayor, en las unidades del CRS métrico. Sin él se
        toma un octavo del lado menor de la extensión.
    iterations : int
        Pasos de simulación. Con pocos quedan solapes, con muchos no cambia.
    anchor_stiffness : float
        Cuánto tira el resorte que sujeta cada círculo a su posición original.
        Subirlo conserva la geografía y deja más solape; bajarlo separa mejor
        los círculos y desarma el mapa.
    damping : float
        Amortiguación del espacio. Sin ella los círculos oscilan y no se
        detienen.
    collision_slop : float
        Penetración que el motor tolera entre dos círculos, en unidades de
        simulación, donde 1 es el radio mayor. **Es el parámetro que decide si
        el cartograma sirve**: con el valor por omisión de pymunk, que es 0,1,
        cada contacto admite un décimo del radio mayor de solape y los círculos
        quedan encimados. Ver las notas.

    Notes
    -----
    La simulación trabaja en unidades del radio mayor y no en las del CRS,
    porque en metros los números son de cinco cifras y el motor pierde
    precisión. Eso obliga a bajar `collision_slop`: sobre las comunas de
    Santiago, con el valor por omisión queda un solape total de 11,4 km, y con
    1e-4 baja a 0,02 km sin que los círculos se alejen más de su lugar.
    """
    import pymunk

    proyectado, original = _projected(geodf)
    valores = proyectado[column].to_numpy(dtype=float)
    if (valores < 0).any():
        raise ValueError(f"'{column}' tiene valores negativos y no se puede escalar por área.")

    x0, y0, x1, y1 = proyectado.total_bounds
    if max_radius is None:
        max_radius = min(x1 - x0, y1 - y0) / 8

    radios = max_radius * np.sqrt(valores / valores.max())
    centroides = projected_centroids(proyectado.geometry)
    centros = np.column_stack([centroides.x.to_numpy(), centroides.y.to_numpy()])

    # La simulación trabaja en unidades del radio mayor. En metros los números
    # son de cinco cifras y el motor pierde precisión.
    escala = max_radius
    origen = centros.mean(axis=0)
    posiciones = (centros - origen) / escala

    space = pymunk.Space()
    space.gravity = (0, 0)
    space.damping = damping
    space.collision_slop = collision_slop

    cuerpos = []
    for (x, y), radio in zip(posiciones, radios / escala):
        masa = max(radio**2, 1e-6)
        cuerpo = pymunk.Body(masa, pymunk.moment_for_circle(masa, 0, radio))
        cuerpo.position = float(x), float(y)
        forma = pymunk.Circle(cuerpo, radio)
        forma.elasticity = 0.2
        space.add(cuerpo, forma)

        # el resorte que lo sujeta a su lugar en el mapa
        ancla = pymunk.Body(body_type=pymunk.Body.STATIC)
        ancla.position = float(x), float(y)
        resorte = pymunk.DampedSpring(
            cuerpo, ancla, (0, 0), (0, 0), 0.0, anchor_stiffness * masa, damping
        )
        space.add(ancla, resorte)
        cuerpos.append(cuerpo)

    for _ in range(iterations):
        space.step(1 / 30)

    finales = np.array([[c.position.x, c.position.y] for c in cuerpos]) * escala + origen

    salida = proyectado.copy()
    salida["geometry"] = [Point(c).buffer(r) for c, r in zip(finales, radios)]
    return salida.to_crs(original) if original else salida


def contiguous_cartogram(geodf, column, iterations=30, tolerance=0.02):
    """Deforma los polígonos hasta que su área sea proporcional a `column`.

    A diferencia de los otros dos cartogramas, este conserva la contigüidad: las
    áreas siguen pegadas unas a otras y el mapa sigue pareciendo un mapa. A
    cambio las formas se distorsionan, que es de donde viene el efecto.

    Usa el método de hoja de goma de Dougenik, Chrisman y Niemeyer (1985): cada
    área ejerce una fuerza sobre todos los vértices del mapa, hacia afuera si
    necesita crecer y hacia adentro si necesita achicarse, y la fuerza decae con
    la distancia. Se repite hasta que las áreas se acercan a las deseadas.

    Parameters
    ----------
    iterations : int
        Cuántas veces se empujan los vértices. Cada iteración acerca las áreas
        a las deseadas y distorsiona un poco más las formas.
    tolerance : float
        Error de área relativo con que se da por terminado, medido como el
        promedio sobre todas las áreas.

    Returns
    -------
    GeoDataFrame
        Con la geometría deformada y una columna `error_area` con lo que le
        falta a cada área para llegar a su valor deseado.

    Notes
    -----
    El método no garantiza llegar al error cero. Sobre las comunas de Santiago,
    donde la comuna central tiene que crecer siete veces y las periféricas
    encogerse al cuatro por ciento de su área, treinta iteraciones dejan un
    error medio del 2% y uno máximo del 61%, siempre en las que más tienen que
    achicarse: un polígono no puede colapsar sin invertirse. Conviene mirar
    `error_area` antes de publicar el mapa.

    El costo crece con el producto de iteraciones, vértices y áreas. Con
    cuarenta comunas de mil ochocientos vértices, treinta iteraciones toman unos
    seis segundos.
    """
    from shapely.ops import transform as shapely_transform

    proyectado, original = _projected(geodf)
    valores = proyectado[column].to_numpy(dtype=float)
    if (valores <= 0).any():
        raise ValueError(f"'{column}' tiene valores no positivos y no se puede deformar por área.")

    geometrias = list(proyectado.geometry)
    area_total = proyectado.area.sum()
    # el área que le tocaría a cada polígono si el mapa dijera la verdad
    deseadas = area_total * valores / valores.sum()

    for _ in range(iterations):
        actuales = np.array([g.area for g in geometrias])
        centros = np.array([[g.centroid.x, g.centroid.y] for g in geometrias])

        radios = np.sqrt(actuales / np.pi)
        radios_deseados = np.sqrt(deseadas / np.pi)
        # cuánto tiene que crecer o achicarse el radio de cada área
        masas = radios_deseados - radios

        error = np.abs(actuales - deseadas) / deseadas
        if error.mean() < tolerance:
            break

        # las fuerzas se atenúan cuando el mapa todavía está lejos del objetivo,
        # así ninguna iteración lo rompe de una vez
        atenuacion = 1 / (1 + error.mean())

        def empujar(x, y, centros=centros, radios=radios, masas=masas, k=atenuacion):
            puntos = np.column_stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float)])
            delta = puntos[:, None, :] - centros[None, :, :]
            distancia = np.hypot(delta[..., 0], delta[..., 1])
            distancia = np.where(distancia == 0, 1e-9, distancia)

            cerca = distancia < radios
            razon = distancia / radios
            # dentro del radio la fuerza crece suave desde el centro; fuera decae
            fuerza = np.where(cerca, masas * razon**2 * (4 - 3 * razon), masas * radios / distancia)

            desplazamiento = (fuerza * k / distancia)[..., None] * delta
            movidos = puntos + desplazamiento.sum(axis=1)
            return movidos[:, 0], movidos[:, 1]

        geometrias = [shapely_transform(empujar, g) for g in geometrias]

    salida = proyectado.copy()
    salida["geometry"] = geometrias
    salida["error_area"] = np.abs(np.array([g.area for g in geometrias]) - deseadas) / deseadas
    return salida.to_crs(original) if original else salida


def cartogram_map(
    geodf, column, kind="non_contiguous", cartogram_args=None, ax=None, fig_args=None, **kwargs
):
    """Dibuja un cartograma, o sea un mapa donde el área codifica un valor.

    `kind` acepta `"non_contiguous"`, que escala cada polígono en torno a su
    centroide; `"dorling"`, que reemplaza cada área por un círculo y los separa;
    y `"contiguous"`, que deforma los polígonos sin despegarlos.
    `cartogram_args` va a la función que construye el cartograma (`max_scale`,
    `iterations`, `anchor_stiffness`, según el tipo) y los argumentos que
    sobran van a `GeoDataFrame.plot`.

    Retorna el eje y el GeoDataFrame deformado, que sirve para anotar encima.
    """
    constructores = {
        "non_contiguous": non_contiguous_cartogram,
        "dorling": dorling_cartogram,
        "contiguous": contiguous_cartogram,
    }
    if kind not in constructores:
        raise ValueError(f"kind debe ser uno de {sorted(constructores)}, no {kind!r}")

    deformado = constructores[kind](geodf, column, **(cartogram_args or {}))

    if ax is None:
        # el encuadre sale de la geometría original, que es la que da el contexto
        fig, ax = figure_from_geodataframe(geodf, **(fig_args or {}))

    deformado.plot(ax=ax, **kwargs)
    return ax, deformado
