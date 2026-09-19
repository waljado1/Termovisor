"""
config.py — Configuración central del proyecto.

Todas las rutas, constantes y umbrales viven aquí. Si algún día
cambias carpetas o límites, solo tocas este archivo.
"""
from pathlib import Path

# ── Rutas base ──────────────────────────────────────────────
BASE = Path(__file__).resolve().parent
UPLOADS = BASE / "uploads"
RESULTADOS = BASE / "resultados_procesamiento"
DETECCIONES = RESULTADOS / "detecciones"
INFORMES = RESULTADOS / "informes"

# Crear carpetas al importar
for _p in (UPLOADS, DETECCIONES, INFORMES):
    _p.mkdir(parents=True, exist_ok=True)

# ── Límites de subida ──────────────────────────────────────
MAX_MB = 40
EXT_OK = {".jpg", ".jpeg", ".png"}
MAX_FILES_PODA = 400          # máximo de archivos por carpeta de salida

# ── Colores por clasificación (hex) ────────────────────────
COLOR_SEV = {
    "CRÍTICO": "#e53935",
    "ALTO": "#fb8c00",
    "MEDIO": "#fdd835",
    "BAJO": "#43a047",
    "HUMEDAD PROBABLE": "#1e88e5",
    "POSIBLE HUMEDAD": "#26c6da",
    "REVISAR": "#9e9e9e",
}

# ── Umbrales de clasificación ──────────────────────────────
UMBRAL_PUENTE = {"CRÍTICO": 3.0, "ALTO": 1.5, "MEDIO": 0.7}   # °C/m
UMBRAL_HUMEDAD = {"PROBABLE": 0.75, "POSIBLE": 0.5}           # probabilidad