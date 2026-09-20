"""
core.ai_analysis.py — Análisis IA.

NIVEL 1: K-means isotermas (clustering de zonas térmicas homogéneas).
NIVEL 2: SAM/MobileSAM — segmentación interactiva por punto.
"""
import cv2
import numpy as np

# ─────────────── NIVEL 1: K-MEANS ISOTERMAS ───────────────
PALETA = np.array([
    [40, 40, 200], [60, 120, 220], [80, 200, 240],
    [120, 220, 180], [60, 160, 90], [30, 30, 180],
    [0, 0, 160], [255, 255, 0],
], dtype=np.uint8)


def kmeans_segmentar(t: np.ndarray, k: int = 5):
    """Agrupa píxeles en k zonas isotermas (clúster 0 = más frío)."""
    from sklearn.cluster import KMeans
    X = t.reshape(-1, 1).astype(np.float32)
    km = KMeans(n_clusters=k, n_init=4, random_state=0).fit(X)
    labels = km.labels_.reshape(t.shape)
    centros = km.cluster_centers_.flatten()
    orden = np.argsort(centros)
    remap = {old: new for new, old in enumerate(orden)}
    labels = np.vectorize(remap.get)(labels).astype(np.int32)
    return labels, np.sort(centros)


def colorizar_clusters(labels: np.ndarray) -> np.ndarray:
    """Pinta cada clúster con un color fijo de la paleta (frío→calor)."""
    img = PALETA[np.clip(labels, 0, len(PALETA) - 1)]
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def analizar_cluster_frio(t: np.ndarray, labels: np.ndarray,
                          gsd_m: float | None) -> dict | None:
    """Datos del clúster más frío con área significativa (1%-35%)."""
    ids, counts = np.unique(labels, return_counts=True)
    total = labels.size
    mejor = None
    for cid, n in zip(ids, counts):
        frac = n / total
        if frac < 0.01 or frac > 0.35:
            continue
        mask = labels == cid
        t_med = float(np.nanmean(t[mask]))
        if mejor is None or t_med < mejor["temp"]:
            mejor = {
                "temp": t_med,
                "frac": frac,
                "area_m2": (round(n * gsd_m * gsd_m, 2) if gsd_m else None),
            }
    return mejor


# ─────────────── NIVEL 2: SAM (segmentación por punto) ───────────────
_sam = {"model": None}


def sam_disponible() -> bool:
    """True si ultralytics está instalado en este entorno."""
    try:
        from ultralytics import SAM  # noqa: F401
        return True
    except ImportError:
        return False


def sam_segmentar(ruta_imagen: str, x: int, y: int) -> np.ndarray | None:
    """
    Segmenta la zona alrededor del punto (x, y) con MobileSAM.
    Devuelve máscara booleana (o None).
    La 1ª llamada descarga mobile_sam.pt (~40 MB) y carga el modelo.
    Ejecutar en hilo aparte (run.io_bound) desde la UI.
    """
    from ultralytics import SAM
    if _sam["model"] is None:
        _sam["model"] = SAM("mobile_sam.pt")
    res = _sam["model"].predict(ruta_imagen, points=[[x, y]],
                                labels=[1], verbose=False)
    if not res or res[0].masks is None:
        return None
    return res[0].masks.data[0].cpu().numpy()


def stats_segmento(t: np.ndarray, mascara: np.ndarray,
                   gsd_m: float | None) -> dict | None:
    """
    Cruza la máscara de SAM con la matriz de temperaturas:
    área (px y m²), T media, desviación y ΔT vs resto de la escena.
    """
    m = cv2.resize(mascara.astype(np.uint8),
                   (t.shape[1], t.shape[0]),
                   interpolation=cv2.INTER_NEAREST).astype(bool)
    if m.sum() < 20:
        return None
    t_seg = float(np.nanmean(t[m]))
    t_resto = float(np.nanmean(t[~m]))
    return {
        "area_px": int(m.sum()),
        "area_m2": round(m.sum() * gsd_m ** 2, 2) if gsd_m else None,
        "temp_media_C": round(t_seg, 2),
        "temp_desv_C": round(float(np.nanstd(t[m])), 2),
        "delta_vs_resto_C": round(t_seg - t_resto, 2),
    }