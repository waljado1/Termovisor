"""
core.imaging.py — Utilidades de imagen térmica.

normalizar(), guardar_termica_colorbar(), guardar_anotada(), podar()
"""
import cv2
import numpy as np
from pathlib import Path

from config import MAX_FILES_PODA


def normalizar(a: np.ndarray) -> np.ndarray:
    """Convierte una matriz float a uint8 0-255 usando percentiles 2-98."""
    a = np.asarray(a, dtype=np.float32)
    validos = np.isfinite(a)
    if not validos.any():
        return np.zeros(a.shape, dtype=np.uint8)
    lo, hi = np.nanpercentile(a[validos], [2, 98])
    if hi <= lo:
        return np.zeros(a.shape, dtype=np.uint8)
    return np.clip((a - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


def guardar_termica_colorbar(t: np.ndarray, path: Path, titulo: str = ""):
    """Térmica PNG con barra de color en °C y rango en el título."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    vmin, vmax = float(np.nanmin(t)), float(np.nanmax(t))
    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=130)
    im = ax.imshow(t, cmap="inferno", vmin=vmin, vmax=vmax)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("°C")
    ax.set_title(f"{titulo}  [{vmin:.1f} – {vmax:.1f} °C]", fontsize=10)
    ax.axis("off")
    fig.savefig(str(path), bbox_inches="tight")
    plt.close(fig)


def guardar_anotada(t, puentes, frias, path: Path):
    """Térmica con rectángulos: amarillo=puentes, cian=zonas frías."""
    img = cv2.applyColorMap(normalizar(t), cv2.COLORMAP_INFERNO)
    for d in puentes:
        cv2.rectangle(img, (d["x"], d["y"]),
                      (d["x"] + d["w"], d["y"] + d["h"]), (0, 255, 255), 2)
    for d in frias:
        cv2.rectangle(img, (d["x"], d["y"]),
                      (d["x"] + d["w"], d["y"] + d["h"]), (255, 255, 0), 2)
    cv2.imwrite(str(path), img)


def podar(carpeta: Path, max_files: int = MAX_FILES_PODA):
    """Borra los archivos más antiguos si se supera el máximo."""
    try:
        files = sorted(carpeta.glob("*"), key=lambda p: p.stat().st_mtime)
        for p in files[:max(0, len(files) - max_files)]:
            p.unlink()
    except OSError:
        pass