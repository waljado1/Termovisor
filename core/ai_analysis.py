"""
core.ai_analysis.py — Análisis IA (Nivel 1: clustering isotermal).

kmeans_segmentar()     -> mapa de zonas homogéneas + centros °C
colorizar_clusters()   -> PNG con paleta frío->calor
analizar_cluster_frio()-> candidato a humedad del clúster más frío
"""
import cv2
import numpy as np

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
    """Pinta cada clúster con un color fijo de la paleta."""
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