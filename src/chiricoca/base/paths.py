import inspect
import sys
from pathlib import Path


def add_project_root(marker: str = "config.py") -> Path:
    """Encuentra la raiz del proyecto y la inserta en sys.path.

    Camina hacia arriba desde el archivo del caller (cuando __file__ esta
    definido, modo script) o desde Path.cwd() (modo REPL/notebook, cuando
    __file__ no existe) buscando un directorio que contenga `marker`.
    Devuelve el Path de la raiz encontrada.

    Util para proyectos donde los scripts viven en subcarpetas y necesitan
    importar modulos del directorio raiz sin instalar el proyecto como
    paquete.
    """
    caller_frame = inspect.stack()[1].frame
    caller_file = caller_frame.f_globals.get("__file__")
    start = (
        Path(caller_file).resolve().parent
        if caller_file
        else Path.cwd().resolve()
    )
    for candidate in [start, *start.parents]:
        if (candidate / marker).exists():
            sys.path.insert(0, str(candidate))
            return candidate
    raise RuntimeError(
        f"No se encontro {marker!r} en ningun directorio padre de {start}"
    )
