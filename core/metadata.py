"""
core.metadata.py — Lectura de metadatos con ExifTool.

Funciones:
    encontrar_exiftool()   -> localiza el binario (cacheado)
    leer_metadata(path)    -> dict completo EXIF/FLIR/DJI en JSON
"""
import json
import subprocess
from pathlib import Path

from config import BASE

_EXIFTOOL: dict = {"cmd": None, "ver": None}


def encontrar_exiftool() -> str | None:
    """Devuelve el comando de exiftool, buscándolo una sola vez."""
    if _EXIFTOOL["cmd"]:
        return _EXIFTOOL["cmd"]
    candidatos = ["exiftool", "exiftool.exe",
                  str(BASE / "exiftool.exe"), str(BASE / "tools" / "exiftool.exe")]
    for c in candidatos:
        try:
            r = subprocess.run(
                [c, "-ver"], capture_output=True, text=True, timeout=5,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if r.returncode == 0 and r.stdout.strip():
                _EXIFTOOL["cmd"] = c
                _EXIFTOOL["ver"] = r.stdout.strip()
                return c
        except Exception:
            continue
    return None


def leer_metadata(path: Path) -> dict:
    """Metadata completa (EXIF + fabricante) vía `exiftool -j`."""
    tool = encontrar_exiftool()
    if not tool:
        return {}
    try:
        r = subprocess.run(
            [tool, "-j", "-struct", str(path)],
            capture_output=True, text=True, timeout=30,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        data = json.loads(r.stdout or "[]")
        return data[0] if isinstance(data, list) and data else {}
    except Exception:
        return {}