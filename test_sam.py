"""test_sam.py — Prueba rápida de MobileSAM (1ª vez descarga mobile_sam.pt)."""
import time
import numpy as np
from ultralytics import SAM

print("Cargando MobileSAM (1ª vez descarga ~40 MB)...")
t0 = time.time()
modelo = SAM("mobile_sam.pt")
print(f"Modelo listo en {time.time()-t0:.1f}s")

# Imagen de prueba sintética (un cuadrado "mancha" sobre fondo)
img = np.zeros((640, 640, 3), dtype=np.uint8)
img[200:400, 250:450] = (0, 0, 220)   # "mancha" azul (BGR)

print("Segmentando punto (320, 300)...")
t0 = time.time()
res = modelo.predict(img, points=[[320, 300]], labels=[1], verbose=False)
dt = time.time() - t0

m = res[0].masks.data[0].cpu().numpy()
print(f"Segmentado en {dt:.1f}s")
print(f"Área de máscara: {int(m.sum())} px (esperado ~40.000)")
print("✅ SAM FUNCIONA" if m.sum() > 20000 else "⚠️ Máscara sospechosa")