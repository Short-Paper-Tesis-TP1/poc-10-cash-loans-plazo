"""
PoC de verificación del plazo: ¿el salto de plazo_meses se mantiene solo en "Cash loans"?
=========================================================================================
Motivo: plazo_meses = monto / cuota. En Home Credit la cuota incluye la tasa que la propia financiera asignó
según su evaluación de riesgo, así que parte del salto (+0.047 de AUC) podría ser esa evaluación interna y no
el plazo pedido. Cash loans (préstamos en cuotas) es el producto que refleja el formulario.

Compara, con todas las mujeres y solo con Cash loans, cada configuración con la desviación estándar entre folds:
  A2 BASE, 4 restricciones SBS | B4 A2 + plazo_meses | C6 A2 + años trabajo/negocio + plazo_meses

Uso: python poc_cash_loans.py "C:\\ruta\\home-credit-default-risk"
"""
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

RUTA = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\alexa\Escritorio\UPC\TP1\home-credit-default-risk"
SEMILLA = 42
EDUCACION = {"Lower secondary": 0, "Secondary / secondary special": 1, "Incomplete higher": 2,
             "Higher education": 3, "Academic degree": 3}


def csv(nombre):
    return os.path.join(RUTA, f"dataset-{nombre}.csv")


t0 = time.time()
app = pd.read_csv(csv("application_train"), usecols=[
    "SK_ID_CURR", "TARGET", "CODE_GENDER", "AMT_INCOME_TOTAL", "AMT_ANNUITY", "AMT_CREDIT", "DAYS_BIRTH",
    "NAME_EDUCATION_TYPE", "CNT_CHILDREN", "NAME_INCOME_TYPE", "ORGANIZATION_TYPE", "DAYS_EMPLOYED",
    "NAME_CONTRACT_TYPE"])
w = app[app.CODE_GENDER == "F"].reset_index(drop=True)

bur = pd.read_csv(csv("bureau"), usecols=["SK_ID_CURR", "SK_ID_BUREAU", "CREDIT_ACTIVE", "AMT_CREDIT_SUM_DEBT"])
en_buro = w.SK_ID_CURR.isin(bur.SK_ID_CURR)
act = (bur[bur.CREDIT_ACTIVE == "Active"].assign(d=lambda x: x.AMT_CREDIT_SUM_DEBT.fillna(0))
       .groupby("SK_ID_CURR").agg(n=("SK_ID_BUREAU", "count"), deuda=("d", "sum")))
w = w.merge(act, left_on="SK_ID_CURR", right_index=True, how="left")

bb = pd.read_csv(csv("bureau_balance"), usecols=["SK_ID_BUREAU", "STATUS"],
                 dtype={"SK_ID_BUREAU": "int64", "STATUS": "category"})
ids_bb = bb.SK_ID_BUREAU.unique()
ids_atr = bb.loc[bb.STATUS.isin(["1", "2", "3", "4", "5"]), "SK_ID_BUREAU"].unique()
del bb
c_bb = bur.loc[bur.SK_ID_BUREAU.isin(ids_bb), "SK_ID_CURR"].unique()
c_atr = bur.loc[bur.SK_ID_BUREAU.isin(ids_atr), "SK_ID_CURR"].unique()

V = pd.DataFrame(index=w.index)
V["ratio_cuota_ingreso"] = w.AMT_ANNUITY / w.AMT_INCOME_TOTAL
V["ratio_deuda_ingreso"] = (w.AMT_CREDIT + np.where(en_buro, w.deuda.fillna(0), np.nan)) / w.AMT_INCOME_TOTAL
V["ratio_monto_ingreso"] = w.AMT_CREDIT / w.AMT_INCOME_TOTAL
V["n_otros_creditos"] = np.where(en_buro, w.n.fillna(0), np.nan)
V["atraso_historial"] = np.where(w.SK_ID_CURR.isin(c_atr), 1.0, np.where(w.SK_ID_CURR.isin(c_bb), 0.0, np.nan))
V["edad"] = -w.DAYS_BIRTH / 365.25
V["nivel_educativo"] = w.NAME_EDUCATION_TYPE.map(EDUCACION)
V["n_hijos"] = w.CNT_CHILDREN
V["anios_trabajo_negocio"] = -w.DAYS_EMPLOYED.where(w.DAYS_EMPLOYED != 365243) / 365.25
V["plazo_meses"] = w.AMT_CREDIT / w.AMT_ANNUITY

BASE = ["ratio_cuota_ingreso", "ratio_deuda_ingreso", "ratio_monto_ingreso", "n_otros_creditos",
        "atraso_historial", "edad", "nivel_educativo", "n_hijos"]
SBS = {"ratio_cuota_ingreso", "ratio_deuda_ingreso", "n_otros_creditos", "atraso_historial"}
y_all = w.TARGET.values
emp_all = ((w.NAME_INCOME_TYPE == "Commercial associate") | (w.ORGANIZATION_TYPE == "Self-employed")).values
cash_all = (w.NAME_CONTRACT_TYPE == "Cash loans").values


def auc_cv(cols, filas):
    X, y, emp = V.loc[filas, cols].reset_index(drop=True), y_all[filas], emp_all[filas]
    mono = [1 if c in SBS else 0 for c in cols]
    todas, empr = [], []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEMILLA).split(X, y):
        m = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=200,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.9,
                               monotone_constraints=mono, random_state=SEMILLA, verbose=-1)
        m.fit(X.iloc[tr], y[tr])
        p = m.predict_proba(X.iloc[te])[:, 1]
        todas.append(roc_auc_score(y[te], p))
        empr.append(roc_auc_score(y[te][emp[te]], p[emp[te]]))
    return {"auc_todas": float(np.mean(todas)), "sd_todas": float(np.std(todas)),
            "auc_emprendedoras": float(np.mean(empr)), "sd_emprendedoras": float(np.std(empr))}


configs = {"A2 BASE, 4 restricciones SBS": BASE,
           "B4 A2 + plazo": BASE + ["plazo_meses"],
           "C6 A2 + trabajo + plazo": BASE + ["anios_trabajo_negocio", "plazo_meses"]}
poblaciones = {"Todos los préstamos": np.ones(len(w), dtype=bool),
               "Solo Cash loans": cash_all,
               "Solo Revolving loans": ~cash_all}

res = {}
for pob, filas in poblaciones.items():
    res[pob] = {"filas": int(filas.sum()), "emprendedoras": int(emp_all[filas].sum()),
                "tasa_target": float(y_all[filas].mean()), "configs": {}}
    for nombre, cols in configs.items():
        res[pob]["configs"][nombre] = auc_cv(cols, filas)
    ref = res[pob]["configs"]["A2 BASE, 4 restricciones SBS"]
    for r in res[pob]["configs"].values():
        r["vs_A2_todas"] = r["auc_todas"] - ref["auc_todas"]
        r["vs_A2_emprendedoras"] = r["auc_emprendedoras"] - ref["auc_emprendedoras"]

print(f"\nTotal mujeres: {len(w):,} | {time.time()-t0:.0f} s")
for pob, d in res.items():
    print(f"\n== {pob}: {d['filas']:,} filas | {d['emprendedoras']:,} emprendedoras | TARGET=1 {d['tasa_target']*100:.2f} %")
    print(f"{'Configuración':30s} {'AUC todas':>9s} {'±sd':>7s} {'vs A2':>7s} {'AUC empr':>9s} {'±sd':>7s} {'vs A2':>7s}")
    for k, r in d["configs"].items():
        print(f"{k:30s} {r['auc_todas']:9.4f} {r['sd_todas']:7.4f} {r['vs_A2_todas']:+7.4f} "
              f"{r['auc_emprendedoras']:9.4f} {r['sd_emprendedoras']:7.4f} {r['vs_A2_emprendedoras']:+7.4f}")
with open("resultado_poc_cash_loans.json", "w", encoding="utf-8") as fh:
    json.dump(res, fh, indent=2, ensure_ascii=False)
print("\nGuardado: resultado_poc_cash_loans.json")
