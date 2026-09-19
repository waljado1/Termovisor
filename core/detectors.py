"""
core.detectors.py — Detectores clásicos de anomalías.

detectar_puentes(): gradientes Sobel en °C/m físicos -> severidad
detectar_frias():   zonas bajo media-k*sigma -> probabilidad de humedad
"""
import cv2
import numpy as np

from config import UMBRAL_PUENTE, UMBRAL_HUMEDAD


def calcular_gsd(md: dict, t: np.ndarray) -> float | None:
    """Metros por píxel a partir de FOV y distancia del metadata."""
    try:
        fov = float(str(md.get("FieldOfView", "")).split()[0])
        dist = float(str(md.get("ObjectDistance", "")).split()[0])
        ancho_m = 2.0 * dist * np.tan(np.radians(fov / 2.0))
        return ancho_m / t.shape[1]
    except Exception:
        return None


def sev_puente(grad_c_m: float | None) -> str:
    """Clasifica un puente térmico por su gradiente en °C/m."""
    if grad_c_m is None:
        return "BAJO"
    if grad_c_m >= UMBRAL_PUENTE["CRÍTICO"]:
        return "CRÍTICO"
    if grad_c_m >= UMBRAL_PUENTE["ALTO"]:
        return "ALTO"
    if grad_c_m >= UMBRAL_PUENTE["MEDIO"]:
        return "MEDIO"
    return "BAJO"


def clasif_fria(delta: float, area: float) -> tuple[float, str]:
    """Probabilidad de humedad (0-1) y etiqueta según ΔT y área."""
    prob = min(0.6 * min(abs(delta) / 2.0, 1.0)
               + 0.4 * min(area / 300.0, 1.0), 1.0)
    if prob >= UMBRAL_HUMEDAD["PROBABLE"]:
        return round(prob, 2), "HUMEDAD PROBABLE"
    if prob >= UMBRAL_HUMEDAD["POSIBLE"]:
        return round(prob, 2), "POSIBLE HUMEDAD"
    return round(prob, 2), "REVISAR"


def detectar_puentes(t: np.ndarray, cfg: dict, gsd: float | None):
    """Puentes térmicos: contornos de alto gradiente (°C/px reales)."""
    blur = cv2.GaussianBlur(t, (5, 5), 0)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)

    umbral = float(np.nanpercentile(grad, cfg["pct_grad"]))
    mask = (grad >= umbral).astype(np.uint8) * 255
    k = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < cfg["area_min_px"]:
            continue
        x, y, w, h = cv2.boundingRect(c)
        m = mask[y:y + h, x:x + w] > 0
        g_px = float(np.mean(grad[y:y + h, x:x + w][m]))
        g_m = g_px / gsd if gsd else None
        out.append({
            "tipo": "Puente térmico", "x": x, "y": y, "w": w, "h": h,
            "area_px": round(float(area), 1),
            "temp_media_C": round(float(np.nanmean(t[y:y + h, x:x + w])), 2),
            "grad_C_m": round(g_m, 2) if g_m is not None else None,
            "severidad": sev_puente(g_m),
        })
    return out, mask


def detectar_frias(t: np.ndarray, cfg: dict):
    """Zonas frías: píxeles bajo media - k*desviación, morfología y contornos."""
    validos = np.isfinite(t)
    if not validos.any():
        return [], np.zeros(t.shape, dtype=np.uint8)
    media = float(np.nanmean(t[validos]))
    desvio = float(np.nanstd(t[validos]))
    limite = media - max(cfg["k_fria"] * desvio, 0.5)
    mask = ((t <= limite) & validos).astype(np.uint8) * 255

    k = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < cfg["area_min_fria_px"]:
            continue
        x, y, w, h = cv2.boundingRect(c)
        temp = float(np.nanmean(t[y:y + h, x:x + w]))
        delta = round(media - temp, 2)
        prob, etiq = clasif_fria(delta, area)
        out.append({
            "tipo": "Zona fría", "x": x, "y": y, "w": w, "h": h,
            "area_px": round(float(area), 1),
            "temp_media_C": round(temp, 2),
            "delta_media_C": delta,
            "probabilidad": prob,
            "severidad": etiq,
        })
    return out, mask