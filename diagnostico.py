"""diagnostico.py — Verificación del entorno antes de arrancar."""
import sys
print("Python:", sys.version)
for name in ["PIL", "cv2", "numpy", "pandas", "plotly", "matplotlib",
             "sklearn", "nicegui", "flirimageextractor"]:
    try:
        m = __import__(name)
        print(f"{name:22s} OK  {getattr(m, '__version__', '')}")
    except Exception as e:
        print(f"{name:22s} ERROR  {e}")