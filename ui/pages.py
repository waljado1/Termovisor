"""
ui.pages.py — Página principal de la aplicación.

IMPORTANTE: los handlers se definen ANTES de construir la UI
(para evitar UnboundLocalError con las referencias).
"""
import asyncio
import json
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from nicegui import ui, run

from config import UPLOADS, DETECCIONES, INFORMES, MAX_MB, EXT_OK
from core.pipeline import procesar, crear_zip
from core.ai_analysis import (kmeans_segmentar, colorizar_clusters,
                              analizar_cluster_frio)
from core.metadata import encontrar_exiftool, _EXIFTOOL
from core.extractor import FLIR_OK, _modulo_dji
from ui.components import (js_descargar, tabla_detecciones, mapa_plotly,
                           termometro_interactivo, sam_panel)

import sys


@ui.page("/")
def index():
    # ══════════ 1) ESTADO ══════════
    cfg = {"pct_grad": 85.0, "area_min_px": 50.0,
           "area_min_fria_px": 80.0, "k_fria": 0.8}
    resultados: list[dict] = []
    lock = asyncio.Lock()
    ultimo = {"path": None}

    # ══════════ 2) HANDLERS ══════════
    def render_resultado(r: dict):
        with area:
            with ui.expansion(f"{'✅' if r['radiometria_ok'] else '⚠️'} "
                              f"[{r.get('motor', '?')}] {r['archivo']}"
                              f" — {r['marca']}",
                              icon="thermostat").classes("w-full"):
                if not r["radiometria_ok"]:
                    ui.label("RADIOMETRÍA NO DISPONIBLE").classes(
                        "text-h6 text-negative")
                    ui.code(r["error"] or "Error").style(
                        "white-space:pre-wrap;max-height:300px;overflow:auto")
                    return

                t = r["thermal"]
                ui.label(f"RADIOMETRÍA {r.get('motor', '')}: OK"
                         ).classes("text-h6 text-positive")
                with ui.row():
                    ui.badge(f"Min {r['temp_min']:.1f} °C", color="blue-8")
                    ui.badge(f"Media {r['temp_mean']:.1f} °C", color="teal-8")
                    ui.badge(f"Máx {r['temp_max']:.1f} °C", color="red-8")
                    if r["gsd_cm_px"]:
                        ui.badge(f"{r['gsd_cm_px']} cm/px", color="grey-8")

                p = r["paths"]
                with ui.row().classes("w-full flex-wrap"):
                    ui.image(f"/detecciones/{p['colorbar']}").classes("w-[420px]")
                    ui.image(f"/detecciones/{p['anotada']}").classes("w-[420px]")
                    if "fusion" in p:
                        ui.image(f"/detecciones/{p['fusion']}").classes("w-[420px]")

                ui.separator()
                ui.label(f"PUENTES TÉRMICOS ({len(r['puentes'])})"
                         ).classes("text-h6 text-red-4")
                tabla_detecciones(r["puentes"], ["area", "temp", "grad"],
                                  r, "PUENTE")

                ui.separator()
                ui.label(f"ZONAS FRÍAS / HUMEDADES ({len(r['frias'])})"
                         ).classes("text-h6 text-blue-4")
                tabla_detecciones(r["frias"], ["area", "temp", "dT", "prob"],
                                  r, "FRIA")

                ui.separator()
                with ui.card().classes("w-full"):
                    ui.label("TERMÓMETRO INTERACTIVO").classes("text-h6")
                    termometro_interactivo(r)

                # ── 🎯 SAM: segmentación por punto ──
                with ui.expansion("🎯 SAM — Segmentación con IA",
                                  icon="center_focus_strong").classes("w-full"):
                    sam_panel(r)

                # ── 🤖 IA: K-MEANS ──
                with ui.expansion("🤖 ANÁLISIS IA — Zonas isotermas",
                                  icon="psychology").classes("w-full"):
                    cont_ia = ui.column().classes("w-full")
                    k_sel = ui.slider(min=3, max=8, value=5,
                                      step=1).props("label-always")

                    def _ejecutar_ia():
                        cont_ia.clear()
                        k = int(k_sel.value)
                        labels, centros = kmeans_segmentar(t, k)
                        fn_ia = DETECCIONES / f"{r['stem']}_kmeans.png"
                        cv2.imwrite(str(fn_ia), colorizar_clusters(labels))
                        gsd_m = ((r["gsd_cm_px"] / 100.0)
                                 if r.get("gsd_cm_px") else None)
                        frio = analizar_cluster_frio(t, labels, gsd_m)
                        with cont_ia:
                            ui.image(f"/detecciones/{fn_ia.name}"
                                     ).classes("w-[420px]")
                            if frio:
                                txt = (f"❄️ Zona más fría: "
                                       f"{frio['temp']:.2f} °C · "
                                       f"{frio['frac']*100:.1f}% de la imagen")
                                if frio["area_m2"]:
                                    txt += f" · ≈ {frio['area_m2']} m²"
                                ui.label(txt).classes(
                                    "text-subtitle1 text-blue-4")
                                ui.label("Candidato a humedad — verificar"
                                         ).classes("text-caption text-orange-5")
                            else:
                                ui.label("Sin clúster frío significativo"
                                         ).classes("text-caption")
                            ui.label("Centros: " + " · ".join(
                                f"{c:.1f}°C" for c in centros
                            )).classes("text-caption")

                    with ui.row().classes("w-full items-center"):
                        ui.label("Nº de zonas (k):").classes("text-caption")
                        ui.button("ANALIZAR CON IA", icon="psychology",
                                  on_click=_ejecutar_ia
                                  ).props("unelevated color=primary")

                with ui.expansion("🗺️ Mapa interactivo").classes("w-full"):
                    mapa_plotly(t)

                with ui.expansion("📥 Descargas").classes("w-full"):
                    with ui.row():
                        ui.button("JSON", icon="download", on_click=lambda:
                                  js_descargar(f"/informes/{p['json']}",
                                               p["json"])).props("dense outline")
                        ui.button("CSV", icon="download", on_click=lambda:
                                  js_descargar(f"/informes/{p['csv']}",
                                               p["csv"])).props("dense outline")
                        ui.button("ZIP todo", icon="archive", on_click=lambda:
                                  _zip(r)).props("dense unelevated")

                with ui.expansion("METADATOS").classes("w-full"):
                    ui.code(json.dumps(r["metadata"], ensure_ascii=False,
                                       indent=2, default=str)
                            ).style("max-height:500px;overflow:auto")

    async def _zip(r: dict):
        zp = await run.cpu_bound(crear_zip, r)
        if zp:
            js_descargar(f"/informes/{zp.name}", zp.name)

    def _actualizar_lote():
        filas = []
        for r in resultados:
            base = {"archivo": r["archivo"], "motor": r.get("motor", "?"),
                    "hora": r["marca"]}
            if not r["radiometria_ok"]:
                filas.append({**base, "t_min": "—", "t_med": "—",
                              "t_max": "—", "puentes": "—", "frias": "—"})
            else:
                filas.append({**base,
                              "t_min": f"{r['temp_min']:.1f}",
                              "t_med": f"{r['temp_mean']:.1f}",
                              "t_max": f"{r['temp_max']:.1f}",
                              "puentes": len(r["puentes"]),
                              "frias": len(r["frias"])})
        tabla_lote.rows = filas
        tabla_lote.update()

    async def _csv_lote():
        if not resultados:
            ui.notify("No hay resultados", type="warning")
            return
        df = pd.DataFrame([{
            "archivo": r["archivo"], "motor": r.get("motor", "?"),
            "temp_min_C": round(r.get("temp_min", float("nan")), 2),
            "temp_media_C": round(r.get("temp_mean", float("nan")), 2),
            "temp_max_C": round(r.get("temp_max", float("nan")), 2),
            "puentes": len(r.get("puentes", [])),
            "zonas_frias": len(r.get("frias", [])),
        } for r in resultados if r["radiometria_ok"]])
        p = INFORMES / f"lote_{datetime.now():%Y%m%d_%H%M%S}.csv"
        df.to_csv(p, index=False, encoding="utf-8-sig")
        js_descargar(f"/informes/{p.name}", p.name)

    async def _procesar_y_mostrar(path: Path):
        async with lock:
            estado.clear()
            with estado:
                ui.spinner(size="lg")
                ui.label(f"Procesando {path.name}…")
            try:
                r = await run.cpu_bound(procesar, path, dict(cfg))
                resultados.append(r)
                render_resultado(r)
                _actualizar_lote()
                ui.notify("Radiometría OK" if r["radiometria_ok"]
                          else f"Sin radiometría: {r.get('error')}",
                          type="positive" if r["radiometria_ok"] else "warning")
            except Exception as ex:
                ui.notify(f"Error: {type(ex).__name__}: {ex}", type="negative")
            finally:
                estado.clear()

    async def _reanalizar():
        if not ultimo["path"]:
            ui.notify("Aún no hay imágenes", type="warning")
            return
        await _procesar_y_mostrar(ultimo["path"])

    async def cargar(e):
        name = Path(e.file.name).name
        if Path(name).suffix.lower() not in EXT_OK:
            ui.notify("Solo JPG/JPEG/PNG", type="negative")
            return
        data = await e.file.read()
        if len(data) > MAX_MB * 1024 * 1024:
            ui.notify(f"Máximo {MAX_MB} MB", type="negative")
            return
        stored = UPLOADS / f"{uuid.uuid4().hex[:6]}_{name}"
        stored.write_bytes(data)
        ultimo["path"] = stored
        await _procesar_y_mostrar(stored)

    # ══════════ 3) UI ══════════
    ui.label("INSPECCIÓN TERMOGRÁFICA DE FACHADAS").classes("text-h4")
    ui.label("FLIR + DJI · análisis exploratorio · IA isotermas + SAM"
             ).classes("text-subtitle1")

    with ui.card().classes("w-full"):
        ui.label("Entorno").classes("text-h6")
        tool = encontrar_exiftool()
        dji_ok = _modulo_dji() is not None
        ui.label(f"Python {sys.version.split()[0]} · "
                 f"flirimageextractor: {'OK' if FLIR_OK else 'ERROR'} · "
                 f"ExifTool: {_EXIFTOOL['ver'] if tool else 'NO ENCONTRADO'} · "
                 f"Motor DJI: {'OK' if dji_ok else 'no instalado'}")

    with ui.expansion("⚙️ Parámetros de detección",
                      icon="tune").classes("w-full"):
        with ui.row().classes("w-full flex-wrap"):
            ui.number("Percentil gradiente", value=85, min=50, max=99, step=1,
                      on_change=lambda e: cfg.update(
                          pct_grad=float(e.value))).props("dense outlined")
            ui.number("Área mín. puentes (px)", value=50, min=10, step=10,
                      on_change=lambda e: cfg.update(
                          area_min_px=float(e.value))).props("dense outlined")
            ui.number("Área mín. zonas frías (px)", value=80, min=10, step=10,
                      on_change=lambda e: cfg.update(
                          area_min_fria_px=float(e.value))).props("dense outlined")
            ui.number("k desviaciones (frío)", value=0.8, min=0.3, max=3.0,
                      step=0.1,
                      on_change=lambda e: cfg.update(
                          k_fria=float(e.value))).props("dense outlined")

    ui.upload(label="Cargar imágenes (FLIR o DJI)", multiple=True,
              auto_upload=True, on_upload=cargar
              ).props("accept=.jpg,.jpeg,.png").classes("w-full")

    estado = ui.column().classes("w-full")

    with ui.card().classes("w-full"):
        ui.label("RESUMEN DEL LOTE").classes("text-h6")
        tabla_lote = ui.table(
            columns=[{"name": c, "label": l, "field": c, "align": "left"}
                     for c, l in [
                ("archivo", "Archivo"), ("motor", "Motor"), ("hora", "Hora"),
                ("t_min", "T mín"), ("t_med", "T media"), ("t_max", "T máx"),
                ("puentes", "Puentes"), ("frias", "Z. frías")]],
            rows=[], row_key="archivo").classes("w-full")
        ui.button("CSV del lote", icon="download",
                  on_click=_csv_lote).props("outline dense")

    area = ui.column().classes("w-full")

    with ui.expansion("⚙️ Acciones", icon="settings").classes("w-full"):
        ui.button("RE-ANALIZAR última imagen", icon="refresh",
                  on_click=_reanalizar).props("outline dense")

    ui.separator()
    ui.label("Los detectores son exploratorios y no constituyen por sí solos "
             "un diagnóstico de patología.").classes("text-caption")