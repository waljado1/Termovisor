"""
core.pipeline.py — Orquestador: coordina todos los módulos.

procesar(path, cfg) -> dict resultado completo (lo que consume la UI)
crear_zip(r)        -> ZIP con todos los artefactos de una imagen
"""
import json
import uuid
import zipfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from config import DETECCIONES, INFORMES
from core.metadata import leer_metadata
from core.extractor import extraer_termica
from core.imaging import (normalizar, guardar_termica_colorbar,
                          guardar_anotada, podar)
from core.detectors import calcular_gsd, detectar_puentes, detectar_frias


def procesar(path: Path, cfg: dict) -> dict:
    """Pipeline completo para una imagen: extraer -> detectar -> guardar."""
    stem = f"{uuid.uuid4().hex[:6]}_{path.stem}"
    md = leer_metadata(path)
    ex = extraer_termica(path, md)

    r = {
        "archivo": path.name, "stem": stem, "metadata": md,
        "motor": ex.get("motor", "?"),
        "radiometria_ok": ex["ok"], "error": ex.get("error"),
        "marca": datetime.now().strftime("%H:%M:%S"),
        "paths": {},
    }
    if not ex["ok"]:
        return r

    t: np.ndarray = ex["thermal"]
    r["thermal"] = t

    gsd = calcular_gsd(md, t)
    r["gsd_cm_px"] = round(gsd * 100, 2) if gsd else None

    puentes, mp = detectar_puentes(t, cfg, gsd)
    frias, mf = detectar_frias(t, cfg)

    p = r["paths"]
    p["colorbar"] = f"{stem}_termica.png"
    p["raw"] = f"{stem}_raw.png"
    p["anotada"] = f"{stem}_anomalias.png"
    p["mp"] = f"{stem}_puentes.png"
    p["mf"] = f"{stem}_frias.png"

    guardar_termica_colorbar(t, DETECCIONES / p["colorbar"], path.stem)
    cv2.imwrite(str(DETECCIONES / p["raw"]),
                cv2.applyColorMap(normalizar(t), cv2.COLORMAP_INFERNO))
    guardar_anotada(t, puentes, frias, DETECCIONES / p["anotada"])
    cv2.imwrite(str(DETECCIONES / p["mp"]), mp)
    cv2.imwrite(str(DETECCIONES / p["mf"]), mf)

    if ex.get("rgb") is not None:
        try:
            rgb_small = cv2.resize(ex["rgb"], (t.shape[1], t.shape[0]))
            t_col = cv2.applyColorMap(normalizar(t), cv2.COLORMAP_INFERNO)
            fusion = cv2.addWeighted(
                cv2.cvtColor(rgb_small, cv2.COLOR_RGB2BGR), 0.55, t_col, 0.45, 0)
            p["fusion"] = f"{stem}_fusion.png"
            cv2.imwrite(str(DETECCIONES / p["fusion"]), fusion)
        except Exception:
            pass

    vals = t[np.isfinite(t)]
    data = {
        "archivo": path.name, "motor": r["motor"],
        "fecha_analisis": datetime.now().isoformat(timespec="seconds"),
        "gsd_cm_px": r["gsd_cm_px"],
        "temperatura_min_C": float(vals.min()),
        "temperatura_media_C": float(vals.mean()),
        "temperatura_max_C": float(vals.max()),
        "posibles_puentes": len(puentes),
        "posibles_zonas_frias": len(frias),
        "detecciones_puentes": puentes,
        "detecciones_zonas_frias": frias,
        "metadata": md,
    }
    p["json"] = f"{stem}_informe.json"
    p["csv"] = f"{stem}_informe.csv"
    (INFORMES / p["json"]).write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    pd.DataFrame([{
        "archivo": data["archivo"], "motor": r["motor"],
        "temp_min_C": data["temperatura_min_C"],
        "temp_media_C": data["temperatura_media_C"],
        "temp_max_C": data["temperatura_max_C"],
        "puentes": len(puentes), "zonas_frias": len(frias),
    }]).to_csv(INFORMES / p["csv"], index=False, encoding="utf-8-sig")

    r.update({
        "temp_min": float(vals.min()), "temp_mean": float(vals.mean()),
        "temp_max": float(vals.max()),
        "puentes": puentes, "frias": frias,
    })
    podar(DETECCIONES)
    podar(INFORMES)
    return r


def crear_zip(r: dict) -> Path | None:
    """ZIP con todos los artefactos generados para una imagen."""
    zp = INFORMES / f"{r['stem']}_{datetime.now():%Y%m%d_%H%M%S}.zip"
    try:
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
            for nombre in r.get("paths", {}).values():
                f = (INFORMES / nombre) if nombre.endswith((".json", ".csv")) \
                    else (DETECCIONES / nombre)
                if f.exists():
                    z.write(f, f.name)
        return zp
    except Exception:
        return None