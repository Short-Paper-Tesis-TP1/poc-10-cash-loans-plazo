"""
Tasa de dificultades de pago (TARGET = 1) por tramo del plazo calculado (monto / cuota),
en mujeres con prestamos de tipo Cash loans. Sustenta la Fig. 2 del short paper.
Uso: python tasa_por_plazo.py "C:\\ruta\\home-credit-default-risk"
"""
import json, os, sys
import pandas as pd

RUTA = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\alexa\Escritorio\UPC\TP1\home-credit-default-risk"
archivo = next(os.path.join(RUTA, n) for n in ("dataset-application_train.csv", "application_train.csv")
               if os.path.exists(os.path.join(RUTA, n)))
w = pd.read_csv(archivo, usecols=["CODE_GENDER", "NAME_CONTRACT_TYPE", "AMT_CREDIT", "AMT_ANNUITY", "TARGET"])
w = w[(w.CODE_GENDER == "F") & (w.NAME_CONTRACT_TYPE == "Cash loans")].copy()
w["plazo"] = w.AMT_CREDIT / w.AMT_ANNUITY
print(f"Mujeres con Cash loans: {len(w):,} | plazo min {w.plazo.min():.1f} | max {w.plazo.max():.1f} | nulos {w.plazo.isna().sum()}")
tramos = pd.cut(w.plazo, bins=[8, 12, 18, 24, 30, 36, 48], include_lowest=True)
t = w.groupby(tramos, observed=True).TARGET.agg(filas="size", pct_default=lambda s: round(s.mean() * 100, 2))
print(t.to_string())
print(f"Filas con tramo asignado: {int(t.filas.sum()):,} | tasa promedio: {w.loc[tramos.notna(), 'TARGET'].mean()*100:.2f} %")
json.dump({"poblacion": len(w), "tramos": {str(k): v for k, v in t.to_dict("index").items()}},
          open("resultado_tasa_por_plazo.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print("Guardado: resultado_tasa_por_plazo.json")
