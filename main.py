"""
main.py — Punto de entrada de la aplicación.

Local:   python main.py   →   http://127.0.0.1:8080
Render:  usa la variable de entorno PORT automáticamente.
"""
import os

from nicegui import app, ui

from config import UPLOADS, DETECCIONES, INFORMES
from ui.pages import index  # noqa: F401  (registra la ruta "/")


app.add_static_files("/uploads", str(UPLOADS))
app.add_static_files("/detecciones", str(DETECCIONES))
app.add_static_files("/informes", str(INFORMES))

puerto = int(os.environ.get("PORT", os.environ.get("APP_PORT", 7860)))
ui.run(title="Inspección Termográfica", host="0.0.0.0", port=puerto,
       reload=False, show=False, dark=True)