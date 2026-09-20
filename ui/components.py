"""
ui.components.py — Componentes reutilizables de la interfaz.

Patrón de interacción (leer / fijar / actuar):
  1) El cursor sobre la imagen SOLO LEE: muestra posición y temperatura
  2) Un CLIC sobre la imagen (o el botón FIJAR) fija el punto en X/Y
  3) MEDIR / SEGMENTAR usan los X/Y fijados (estables)

Compatible con NiceGUI 3.x: eventos crudos del navegador convertidos
a coords de imagen con JS estándar (querySelector + getBoundingClientRect).
"""
import asyncio
import time

import cv2
import numpy as np
import plotly.graph_objects as go

from nicegui import ui, run

from config import DETECCIONES, COLOR_SEV
from core.imaging import normalizar
from core.ai_analysis import (sam_disponible, sam_segmentar,
                              stats_segmento)

_debug_js = {"err": None}
_JS_CONTADOR = {"n": 0}


# ────────────────────────── Descargas ───────────────────────────────────
def js_descargar(url: str, nombre: str):
    """Dispara la descarga de un archivo en el navegador."""
    ui.run_javascript(
        f'const a=document.createElement("a");a.href="{url}";'
        f'a.download="{nombre}";document.body.appendChild(a);'
        f'a.click();a.remove();')


# ───────────── Clase CSS única para localizar la imagen en JS ───────────
def _asignar_clase_unico(itx) -> str:
    """Añade una clase CSS única al elemento interactive_image."""
    _JS_CONTADOR["n"] += 1
    cls = f"jsxy-{_JS_CONTADOR['n']}"
    itx.classes(cls)
    return cls


# ───────────── Conversión evento navegador → coords de imagen ───────────
async def xy_de_navegador(itx, e, cls: str) -> tuple[float | None,
                                                     float | None]:
    """
    Convierte el evento crudo (PointerEvent) a coords de imagen.
    JS estándar: localiza el contenedor por su clase única, mide el
    <img>, escala por naturalWidth/rect.width y devuelve (x, y).
    """
    args = getattr(e, "args", None)

    # Fallback: versiones antiguas que entregan image_x/image_y
    if not isinstance(args, dict):
        x = getattr(e, "image_x", None)
        y = getattr(e, "image_y", None)
        return (x, y) if x is not None else (None, None)

    cx = args.get("clientX", args.get("x"))
    cy = args.get("clientY", args.get("y"))
    if cx is None or cy is None:
        return None, None

    js = (
        f'const root=document.querySelector(".{cls}");'
        'if(!root) return null;'
        'const img=root.querySelector("img");'
        'if(!img) return null;'
        'const r=img.getBoundingClientRect();'
        'const sx=img.naturalWidth/r.width, sy=img.naturalHeight/r.height;'
        f'return [({float(cx)}-r.left)*sx, ({float(cy)}-r.top)*sy];'
    )
    try:
        res = await ui.run_javascript(js)
        if isinstance(res, (list, tuple)) and len(res) == 2 \
                and res[0] is not None:
            return float(res[0]), float(res[1])
        _debug_js["err"] = f"JS devolvió: {res!r}"
    except Exception as ex:
        _debug_js["err"] = f"{type(ex).__name__}: {ex}"

    return None, None


def _medir_en(t: np.ndarray, x: float, y: float) -> float | None:
    """Temperatura mediana 3x3 centrada en (x, y)."""
    H, W = t.shape
    xi, yi = int(round(x)), int(round(y))
    if not (0 <= xi < W and 0 <= yi < H):
        return None
    return float(np.nanmedian(t[max(0, yi - 1):yi + 2,
                               max(0, xi - 1):xi + 2]))


# ────────────────────────── Detalle de detección ────────────────────────
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
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                cv2.LINE_AA)
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


# ────────────────────────── Tabla de detecciones ────────────────────────
def tabla_detecciones(detecciones: list, cols_extra: list, r: dict,
                      tipo: str):
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


# ────────────────────────── Mapa Plotly ─────────────────────────────────
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


# ════════════════════════════════════════════════════════════════════════
#  OBJETIVO: leer con el cursor / FIJAR con clic o botón / actuar
# ════════════════════════════════════════════════════════════════════════
class _ObjetivoCompartido:
    """
    - mousemove: SOLO lectura (label en vivo + guarda última posición)
    - clic en la imagen o botón FIJAR: fija X/Y en los inputs + chincheta
    - los inputs X/Y no cambian al mover el cursor (solo al fijar)
    """

    def __init__(self, t: np.ndarray, ruta_img: str, cross: str,
                 etiqueta: str):
        self.t = t
        self.H, self.W = t.shape
        self._last = {"t": 0.0}
        self.ultimo = {"x": None, "y": None}    # última pos del cursor
        self.marcas: list[dict] = []            # chinchetas dibujadas
        self.lbl = ui.label(etiqueta).classes("text-caption")
        self.itx = ui.interactive_image(ruta_img,
                                        events=["mousemove", "click"],
                                        cross=cross)
        self.cls = _asignar_clase_unico(self.itx)
        self.itx.on("mousemove", self._on_move)
        self.itx.on("click", self._on_click)
        self.itx.classes("w-[640px] max-w-full")
        with ui.row().classes("items-end"):
            self.xin = ui.number("X", value=self.W // 2, min=0,
                                 max=self.W - 1, step=1
                                 ).props("dense outlined style='width:90px'")
            self.yin = ui.number("Y", value=self.H // 2, min=0,
                                 max=self.H - 1, step=1
                                 ).props("dense outlined style='width:90px'")
            ui.button("FIJAR", icon="push_pin",
                      on_click=self._fijar_ultimo
                      ).props("dense outline")

    # ── dibujo de marcas ──
    def _svg(self) -> str:
        parts = []
        for i, m in enumerate(self.marcas):
            color = m.get("color", "#00e5ff")
            texto = m.get("texto")
            x, y = m["x"], m["y"]

            # círculo de la marca
            parts.append(f'<circle cx="{x}" cy="{y}" r="7" '
                         f'fill="none" stroke="{color}" stroke-width="2"/>')

            if not texto:
                continue

            # alternar posición del texto: arriba / abajo según nº de marca
            arriba = (i % 2 == 0)
            ty = (y - 14) if arriba else (y + 22)
            # si está muy arriba en la imagen, forzar abajo (y viceversa)
            if y < 28:
                ty = y + 22
            elif y > self.H - 28:
                ty = y - 14

            # texto con contorno oscuro para legibilidad sobre cualquier fondo
            # (stroke negro + relleno blanco)
            parts.append(
                f'<text x="{x + 10}" y="{ty}" '
                f'font-size="14" font-weight="bold" '
                f'fill="#ffffff" stroke="#000000" stroke-width="3" '
                f'paint-order="stroke fill" '
                f'style="paint-order:stroke fill">{texto}</text>')
        return "".join(parts)

    def refrescar(self):
        try:
            self.itx.set_content(self._svg())
        except Exception:
            pass

    def fijar(self, x: float, y: float, color: str = "#ffeb3b",
              texto: str | None = None):
        """Fija el punto en los inputs y dibuja la chincheta."""
        self.xin.set_value(int(round(x)))
        self.yin.set_value(int(round(y)))
        self.marcas.append({"x": x, "y": y, "color": color,
                            "texto": texto})
        self.refrescar()

    # ── handlers ──
    async def _on_move(self, e):
        now = time.monotonic()
        if now - self._last["t"] < 0.10:    # throttle ~10 lecturas/s
            return
        self._last["t"] = now
        x, y = await xy_de_navegador(self.itx, e, self.cls)
        if x is None:
            return
        xi, yi = int(round(x)), int(round(y))
        if not (0 <= xi < self.W and 0 <= yi < self.H):
            return
        self.ultimo = {"x": xi, "y": yi}    # solo LECTURA, no toca inputs
        v = _medir_en(self.t, xi, yi)
        self.lbl.set_text(f"👀 Cursor en ({xi},{yi}) → {v:.2f} °C  "
                          f"|  CLIC (o FIJAR) para fijar el punto")

    async def _on_click(self, e):
        x, y = await xy_de_navegador(self.itx, e, self.cls)
        if x is None:
            self.lbl.set_text("[debug] clic sin coords — usa FIJAR "
                              "tras mover el cursor")
            return
        self.fijar(x, y, color="#ffeb3b",
                   texto=f"{int(x)},{int(y)}")
        v = _medir_en(self.t, x, y)
        self.lbl.set_text(f"📌 Fijado ({int(x)},{int(y)}) → {v:.2f} °C  "
                          f"|  ahora pulsa el botón de acción")

    def _fijar_ultimo(self):
        if self.ultimo["x"] is None:
            ui.notify("Mueve primero el cursor sobre la imagen",
                      type="warning")
            return
        self.fijar(self.ultimo["x"], self.ultimo["y"],
                   color="#ffeb3b", texto=f"{self.ultimo['x']},"
                                          f"{self.ultimo['y']}")
        self.lbl.set_text(f"📌 Fijado ({self.ultimo['x']},"
                          f"{self.ultimo['y']})  |  pulsa el botón de acción")


# ────────────────────────── Termómetro ──────────────────────────────────
def termometro_interactivo(r: dict):
    """Fijar punto (clic o FIJAR) → MEDIR marca temperatura estable."""
    t = r["thermal"]
    H, W = t.shape

    obj = _ObjetivoCompartido(
        t, f"/detecciones/{r['paths']['raw']}", "#00e5ff",
        "Mueve el cursor para explorar · CLIC o FIJAR para fijar · MEDIR")

    def _medir():
        x, y = float(obj.xin.value), float(obj.yin.value)
        v = _medir_en(t, x, y)
        if v is None:
            obj.lbl.set_text(f"Coordenadas fuera de rango (0-{W}, 0-{H})")
            return
        obj.marcas.append({"x": x, "y": y, "color": "#00e5ff",
                           "texto": f"{v:.1f} °C"})
        obj.refrescar()
        obj.lbl.set_text(f"✅ Medido ({int(x)},{int(y)}) → {v:.2f} °C")

    with ui.row():
        ui.button("MEDIR", icon="thermostat",
                  on_click=_medir).props("unelevated color=primary")


# ────────────────────────── SAM ─────────────────────────────────────────
def sam_panel(r: dict):
    """Fijar punto (clic o FIJAR) → SEGMENTAR lanza SAM con ese punto."""
    if not sam_disponible():
        ui.label("SAM no instalado en este entorno "
                 "(opcional: pip install ultralytics)"
                 ).classes("text-caption text-orange-5")
        return

    t = r["thermal"]
    H, W = t.shape
    gsd_m = (r["gsd_cm_px"] / 100.0) if r.get("gsd_cm_px") else None
    ruta_img = str(DETECCIONES / r["paths"]["raw"])

    obj = _ObjetivoCompartido(
        t, ruta_img, "#00ff88",
        "Mueve el cursor al centro de la mancha · CLIC o FIJAR · SEGMENTAR")
    resultados = ui.column().classes("w-full")

    def _mostrar_resultado(xi: int, yi: int, mascara):
        s = stats_segmento(t, mascara, gsd_m)
        if s is None:
            obj.lbl.set_text("Segmento demasiado pequeño — prueba el "
                             "centro de la mancha")
            return

        overlay = cv2.applyColorMap(normalizar(t), cv2.COLORMAP_INFERNO)
        m_full = cv2.resize(mascara.astype(np.uint8), (W, H),
                            interpolation=cv2.INTER_NEAREST)
        cnts, _ = cv2.findContours(m_full, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, cnts, -1, (0, 255, 120), 2)
        cv2.circle(overlay, (xi, yi), 5, (0, 255, 120), -1)
        fn = DETECCIONES / f"sam_{r['stem']}_{xi}_{yi}.png"
        cv2.imwrite(str(fn), overlay)

        d = s["delta_vs_resto_C"]
        if d <= -1.5:
            etiq, color = "HUMEDAD PROBABLE", "text-blue-4"
        elif d <= -0.7:
            etiq, color = "POSIBLE HUMEDAD", "text-cyan-5"
        elif d >= 1.5:
            etiq, color = "ZONA CALIENTE (¿fuga térmica?)", "text-red-4"
        else:
            etiq, color = "SIN ANOMALÍA TÉRMICA", "text-grey-5"

        obj.lbl.set_text(f"✅ Segmentado en ({xi},{yi})")
        with resultados:
            ui.image(f"/detecciones/{fn.name}").classes("w-[420px]")
            with ui.row().classes("w-full flex-wrap items-center"):
                area_txt = f"Área {s['area_px']} px"
                if s["area_m2"]:
                    area_txt += f" ≈ {s['area_m2']} m²"
                ui.badge(area_txt, color="grey-8")
                ui.badge(f"T media {s['temp_media_C']} °C", color="teal-8")
                ui.badge(f"ΔT {d:+.2f} °C",
                         color="blue-8" if d < 0 else "red-8")
            ui.label(f"Clasificación: {etiq}").classes(
                f"text-subtitle1 {color}")
            ui.label("Verificar visualmente — contorno generado por SAM"
                     ).classes("text-caption")

    async def _segmentar():
        xi, yi = int(obj.xin.value or 0), int(obj.yin.value or 0)
        if not (0 <= xi < W and 0 <= yi < H):
            obj.lbl.set_text(f"Coordenadas fuera de rango (0-{W}, 0-{H})")
            return
        resultados.clear()
        obj.lbl.set_text(f"⏳ SAM segmentando ({xi},{yi})… "
                         f"la 1ª vez carga el modelo (más lento)")
        try:
            # HILO (no proceso): evita el silencio de multiprocessing
            mascara = await run.io_bound(sam_segmentar, ruta_img, xi, yi)
        except Exception as ex:
            obj.lbl.set_text(f"Error SAM: {type(ex).__name__}: {ex}")
            return
        if mascara is None:
            obj.lbl.set_text("SAM no encontró objeto ahí — prueba el "
                             "centro exacto de la mancha")
            return
        _mostrar_resultado(xi, yi, mascara)

    def _segmentar_manual():
        ui.notify("SAM trabajando… espera unos segundos",
                  type="info", timeout=2000)
        asyncio.get_event_loop().create_task(_segmentar())

    with ui.row():
        ui.button("SEGMENTAR CON IA", icon="center_focus_strong",
                  on_click=_segmentar_manual).props(
                      "unelevated color=primary")