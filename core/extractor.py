"""
core.extractor.py — Motores de extracción radiométrica.

extraer_termica(path) -> {"ok", "thermal", "rgb", "motor", "error"}

Selecciona motor según el fabricante del EXIF:
    FLIR -> flirimageextractor (radiométrico JPG)
    DJI  -> dji-thermal-sdk    (R-JPEG de dron)
"""
import numpy as np
from pathlib import Path

from core.metadata import encontrar_exiftool

try:
    import flirimageextractor
    FLIR_OK, FLIR_ERROR = True, ""
except Exception as e:
    FLIR_OK, FLIR_ERROR = False, str(e)

_DJI_MOD = {"obj": None, "probado": False}


def _modulo_dji():
    """Localiza el wrapper DJI instalado (nombres según versión)."""
    if _DJI_MOD["probado"]:
        return _DJI_MOD["obj"]
    _DJI_MOD["probado"] = True
    for nombre in ("dji_thermal_sdk", "dji_sdk.dji_thermal_sdk",
                   "dji_executables.dji_thermal_sdk", "dji_executables"):
        try:
            _DJI_MOD["obj"] = __import__(nombre, fromlist=["*"])
            return _DJI_MOD["obj"]
        except Exception:
            continue
    return None


def extraer_flir(path: Path) -> dict:
    """Matriz de temperaturas de un JPG radiométrico FLIR."""
    if not FLIR_OK:
        return {"ok": False, "error": f"flirimageextractor: {FLIR_ERROR}"}
    if not encontrar_exiftool():
        return {"ok": False, "error": "ExifTool no encontrado"}
    try:
        ex = flirimageextractor.FlirImageExtractor()
        ex.process_image(str(path))
        thermal = np.asarray(ex.get_thermal_np(), dtype=np.float32)
        if thermal.size == 0:
            raise RuntimeError("matriz térmica vacía")
        try:
            rgb = np.asarray(ex.get_rgb_np())
        except Exception:
            rgb = None
        return {"ok": True, "thermal": thermal, "rgb": rgb}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def extraer_dji(path: Path) -> dict:
    """Matriz de temperaturas de un R-JPEG DJI (API resuelta dinámicamente)."""
    mod = _modulo_dji()
    if mod is None:
        return {"ok": False, "error": "dji-thermal-sdk no instalado"}
    try:
        def fn(*nombres):
            for n in nombres:
                f = getattr(mod, n, None)
                if callable(f):
                    return f
            return None

        f_init = fn("init_env", "init", "setup_env")
        f_create = fn("create_thermal_from_rjpeg", "dirp_create_from_rjpeg",
                      "create_thermal", "create_from_rjpeg")
        f_process = fn("process_threads", "dirp_process", "process")
        f_measure = fn("get_measurement", "dirp_get_measurement",
                       "get_temperature_data", "get_temperatures")
        f_destroy = fn("destroy_thermal", "dirp_destroy", "destroy", "close")

        if not (f_create and f_measure):
            attrs = [a for a in dir(mod) if not a.startswith("_")]
            return {"ok": False, "error": f"API DJI no reconocida: {attrs}"}

        if f_init:
            try:
                f_init()
            except Exception:
                pass

        handle = f_create(str(path))
        if handle is None:
            return {"ok": False, "error": "No es un R-JPEG DJI válido"}
        try:
            if f_process:
                try:
                    f_process(handle, 1)
                except Exception:
                    try:
                        f_process(handle)
                    except Exception:
                        pass
            res = f_measure(handle)
        finally:
            if f_destroy:
                try:
                    f_destroy(handle)
                except Exception:
                    pass

        matriz = None
        if isinstance(res, dict):
            for k in ("temperature", "temperatures", "data", "matrix"):
                if k in res:
                    matriz = np.asarray(res[k], dtype=np.float32)
                    break
        elif isinstance(res, (tuple, list)) and res:
            matriz = np.asarray(res[0], dtype=np.float32)
        else:
            matriz = np.asarray(res, dtype=np.float32)

        if matriz is None or matriz.size == 0:
            return {"ok": False, "error": "SDK DJI sin matriz"}

        med = float(np.nanmedian(matriz))
        if 150 < med < 2000:          # décimas de °C -> °C
            matriz = matriz / 10.0
        return {"ok": True, "thermal": matriz.astype(np.float32), "rgb": None}
    except Exception as e:
        return {"ok": False, "error": f"DJI {type(e).__name__}: {e}"}


def extraer_termica(path: Path, metadata: dict) -> dict:
    """Router: elige motor según fabricante y extrae la matriz."""
    marca = str(metadata.get("Make", "")).lower()
    if "dji" in marca or "hasselblad" in marca:
        out = extraer_dji(path)
        out["motor"] = "DJI"
    else:
        out = extraer_flir(path)
        out["motor"] = "FLIR"
    return out