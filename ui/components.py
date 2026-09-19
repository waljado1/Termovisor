"""
ui.components.py — Componentes reutilizables de la interfaz.

js_descargar(), tabla_detecciones(), ver_detalle(),
mapa_plotly(), termometro_interactivo()
"""
import cv2
import numpy as np
import plotly.graph_objects as go

from nicegui import ui

from config import DETECCIONES, COLOR_SEV
from core.imaging import normalizar


def js_descargar(url: str, nombre: str):
    """Dispara la descarga de un archivo en el navegador."""
    ui.run_javascript(
        f'const a=document.createElement("a");a.href="{url}";'
        f'a.download="{nombre}";document.body.appendChild(a);'
        f'a.click();a.remove();')


def ver_detalle(r: dict, d: dict, tipo: str):
    """Diálogo con zoom de la detección marcada en amarillo."""
    t = r["thermal"]
    x, y, w, h = int(d["x"]), int(d["y"]), int(d["w"]), int(d["h"])
    pad = 18
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(t.shape[1], x + w + pad), min(t.shape[0], y + h + pad)
    crop = t[y0:y1, x0:x1]
    img = cv2.applyColorMap(normalizar(crop), cv2.COLORMAP_INFERNO)
    cv2.rectangle(img, (x - x0, y - y0), (x + w - x0, y + h - y0),
                  (0, 255, 255), 2)
    cv2.putText(img, f"T={d['temp_media_C']:.2f}C", (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    fn = DETECCIONES / f"detalle_{r['stem']}_{tipo}_{x}_{y}.png"
    cv2.imwrite(str(fn), img)

    delta = d.get("delta_media_C")
    extra = (f"ΔT: {delta:+.2f} °C" if delta is not None
             else f"Gradiente: {d.get('grad_C_m', '—')} °C/m")
    with ui.dialog() as dlg, ui.card().classes("w-[900px] max-w-[95vw]"):
        ui.label(f"DETALLE — {tipo}").classes("text-h5")
        ui.image(f"/detecciones/{fn.name}").classes("w-full")
        ui.label(f"Centro ({x + w / 2:.0f}, {y + h / 2:.0f}) | "
                 f"Área {d['area_px']:.0f} px²")
        ui.label(f"T media: {d['temp_media_C']:.2f} °C | {extra}")
        ui.button("CERRAR", on_click=dlg.close).props("unelevated")
    dlg.open()


def _on_ver(e, r, detecciones, tipo):
    row = e.args[0] if isinstance(e.args, list) else e.args
    ver_detalle(r, detecciones[int(row["idx"])], tipo)


def tabla_detecciones(detecciones: list, cols_extra: list, r: dict, tipo: str):
    """Tabla con badges de severidad y botón VER por fila."""
    if not detecciones:
        ui.label("No se detectaron candidatos.").classes("text-caption")
        return
    labels = {"area": "Área px²", "temp": "T °C", "grad": "Grad °C/m",
              "dT": "ΔT °C", "prob": "Prob."}
    cols = [{"name": "idx", "label": "#", "field": "idx", "align": "left"}]
    cols += [{"name": c, "label": labels.get(c, c), "field": c,
              "align": "left"} for c in cols_extra]
    cols += [{"name": "severidad", "label": "Clasificación",
              "field": "severidad", "align": "left"},
             {"name": "accion", "label": "", "field": "accion"}]

    rows = []
    for i, d in enumerate(detecciones):
        row = {"idx": i, "severidad": d["severidad"],
               "color": COLOR_SEV.get(d["severidad"], "#9e9e9e"),
               "area": f"{d['area_px']:.0f}",
               "temp": f"{d['temp_media_C']:.2f}"}
        if "grad_C_m" in d:
            row["grad"] = (f"{d['grad_C_m']}"
                           if d["grad_C_m"] is not None else "—")
        if "delta_media_C" in d:
            row["dT"] = f"{d['delta_media_C']:+.2f}"
            row["prob"] = f"{d['probabilidad']:.2f}"
        rows.append(row)

    table = ui.table(columns=cols, rows=rows, row_key="idx").classes("w-full")
    table.add_slot("body-cell-severidad", """
        <q-td :props="props">
            <q-badge :style="'background:'+props.row.color+';color:#111'">
                {{ props.row.severidad }}</q-badge>
        </q-td>""")
    table.add_slot("body-cell-accion", """
        <q-td :props="props">
            <q-btn dense unelevated icon="visibility" label="VER"
                   style="background:#168dcc;color:white"
                   @click="$parent.$emit('ver', props.row)" />
        </q-td>""")
    table.on("ver", lambda e: _on_ver(e, r, detecciones, tipo))


def mapa_plotly(t: np.ndarray):
    """Heatmap interactivo con zoom y hover en °C reales."""
    vmin, vmax = float(np.nanmin(t)), float(np.nanmax(t))
    fig = go.Figure(go.Heatmap(
        z=t, colorscale="Inferno", zmin=vmin, zmax=vmax,
        colorbar=dict(title="°C"),
        hovertemplate="x:%{x} · y:%{y}<br>%{z:.2f} °C<extra></extra>"))
    fig.update_layout(height=440, margin=dict(l=10, r=10, t=30, b=10),
                      yaxis=dict(autorange="reversed"),
                      title="Mapa térmico interactivo (zoom + hover)")
    ui.plotly(fig).classes("w-full")


def termometro_interactivo(r: dict):
    """Imagen interactiva: hover = temperatura en vivo, clic = punto fijado."""
    t = r["thermal"]
    H, W = t.shape
    marc: list[dict] = []
    lbl = ui.label("Haz clic en la imagen para medir la temperatura del píxel"
                   ).classes("text-caption")

    def medir(x, y):
        xi, yi = int(round(x)), int(round(y))
        if not (0 <= xi < W and 0 <= yi < H):
            return None
        return float(np.nanmedian(t[max(0, yi - 1):yi + 2,
                                   max(0, xi - 1):xi + 2]))

    def extraer_xy(e):
        x = getattr(e, "image_x", None)
        y = getattr(e, "image_y", None)
        if x is None:
            args = getattr(e, "args", None)
            if isinstance(args, dict):
                x, y = args.get("image_x"), args.get("image_y")
        return x, y

    def svg():
        parts = []
        for m in marc:
            parts.append(f'<circle cx="{m["x"]}" cy="{m["y"]}" r="7" '
                         f'fill="none" stroke="#00e5ff" stroke-width="2"/>')
            parts.append(f'<text x="{m["x"] + 10}" y="{m["y"] - 8}" '
                         f'fill="#fff" font-size="15" font-weight="bold">'
                         f'{m["t"]:.1f} °C</text>')
        return "".join(parts)

    def on_move(e):
        x, y = extraer_xy(e)
        if x is None:
            return
        v = medir(x, y)
        lbl.set_text(f"{v:.2f} °C" if v is not None else "—")

    def on_click(e):
        x, y = extraer_xy(e)
        if x is None:
            return
        v = medir(x, y)
        if v is None:
            return
        marc.append({"x": x, "y": y, "t": v})
        lbl.set_text(f"Punto ({int(x)},{int(y)}) → {v:.2f} °C")
        try:
            itx.set_content(svg())
        except Exception:
            pass

    itx = ui.interactive_image(
        f"/detecciones/{r['paths']['raw']}",
        events=["click", "mousemove"], cross="#00e5ff")
    itx.on("mousemove", on_move)
    itx.on("click", on_click)
    itx.classes("w-[640px] max-w-full")