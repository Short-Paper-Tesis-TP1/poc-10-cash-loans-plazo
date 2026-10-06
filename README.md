# PoC 10: población Cash loans y verificación de la variable de plazo

Parte del proyecto de tesis "Plataforma web para la detección temprana del riesgo de sobreendeudamiento en mujeres emprendedoras" (UPC, Taller de Proyecto I, 2026-20).

## Objetivo único
Comprobar dos cosas:
1. Si entrenar solo con préstamos de tipo Cash loans cambia el AUC-ROC.
2. Si el plazo calculado como monto ÷ cuota es una señal real o un artefacto del precio de Home Credit.

## Datos
Home Credit Default Risk (Kaggle, 2018): `application_train`, `bureau` y `bureau_balance`. Los CSV no se incluyen en el repositorio porque las reglas de la competencia no permiten redistribuirlos. Se descargan de https://www.kaggle.com/competitions/home-credit-default-risk/data.

Población: mujeres (`CODE_GENDER == "F"`). El subgrupo emprendedoras son las clientas con `NAME_INCOME_TYPE == "Commercial associate"` u `ORGANIZATION_TYPE == "Self-employed"`, y se reporta por separado.

## Método
- Validación cruzada estratificada de 5 particiones con semilla 42, con desviación estándar entre particiones.
- Configuraciones A2, B4 (A2 + plazo) y C6 (A2 + años de trabajo + plazo) en tres poblaciones: todos los préstamos, solo Cash loans y solo Revolving loans.
- `tasa_por_plazo.py` calcula la tasa de dificultades de pago por tramo de plazo en las mujeres con Cash loans.

## Cómo correrla
```
pip install -r requirements.txt
python poc_cash_loans.py "C:\ruta\home-credit-default-risk"
python tasa_por_plazo.py "C:\ruta\home-credit-default-risk"
```

## Resultado
**Todos los préstamos** (202,448 mujeres, 62,091 emprendedoras, tasa de dificultades de pago 7.00 %)

| Configuración | AUC-ROC todas | DE | AUC-ROC emprendedoras | DE | vs A2 todas | vs A2 emprendedoras |
|---|---|---|---|---|---|---|
| A2 BASE, 4 restricciones SBS | 0.6483 | 0.0029 | 0.6381 | 0.0030 | +0.0000 | +0.0000 |
| B4 A2 + plazo | 0.6952 | 0.0039 | 0.6895 | 0.0051 | +0.0469 | +0.0514 |
| C6 A2 + trabajo + plazo | 0.6994 | 0.0037 | 0.6928 | 0.0060 | +0.0510 | +0.0547 |

**Solo Cash loans** (182,800 mujeres, 55,133 emprendedoras, tasa de dificultades de pago 7.18 %)

| Configuración | AUC-ROC todas | DE | AUC-ROC emprendedoras | DE | vs A2 todas | vs A2 emprendedoras |
|---|---|---|---|---|---|---|
| A2 BASE, 4 restricciones SBS | 0.6471 | 0.0045 | 0.6350 | 0.0115 | +0.0000 | +0.0000 |
| B4 A2 + plazo | 0.6964 | 0.0023 | 0.6884 | 0.0060 | +0.0494 | +0.0534 |
| C6 A2 + trabajo + plazo | 0.6998 | 0.0014 | 0.6913 | 0.0041 | +0.0527 | +0.0562 |

**Solo Revolving loans** (19,648 mujeres, 6,958 emprendedoras, tasa de dificultades de pago 5.31 %)

| Configuración | AUC-ROC todas | DE | AUC-ROC emprendedoras | DE | vs A2 todas | vs A2 emprendedoras |
|---|---|---|---|---|---|---|
| A2 BASE, 4 restricciones SBS | 0.6274 | 0.0064 | 0.6367 | 0.0098 | +0.0000 | +0.0000 |
| B4 A2 + plazo | 0.6274 | 0.0064 | 0.6367 | 0.0098 | +0.0000 | +0.0000 |
| C6 A2 + trabajo + plazo | 0.6358 | 0.0196 | 0.6356 | 0.0162 | +0.0084 | -0.0011 |

**Tasa de dificultades de pago por tramo de plazo (monto ÷ cuota), mujeres con Cash loans (182,800)**

| Tramo de plazo | Filas | Tasa (%) |
|---|---|---|
| (7.999, 12.0] | 21,694 | 3.92 |
| (12.0, 18.0] | 38,249 | 8.33 |
| (18.0, 24.0] | 53,416 | 9.06 |
| (24.0, 30.0] | 25,158 | 6.25 |
| (30.0, 36.0] | 36,380 | 6.40 |
| (36.0, 48.0] | 7,895 | 4.40 |

## Decisión de diseño
- El modelo se entrena solo con Cash loans, porque es coherente con el formulario y no cuesta AUC-ROC.
- El plazo se excluye: la tasa de dificultades de pago no es monótona respecto del plazo y sus valores son continuos. Esto indica que el cociente monto ÷ cuota separa tramos de precio de Home Credit y no plazos elegidos por la clienta.

## Archivos
- `poc_cash_loans.py`: script principal.
- `tasa_por_plazo.py`: tabla de tasa por tramo de plazo.
- `resultado_poc_cash_loans.json` y `resultado_tasa_por_plazo.json`: salidas estructuradas.
- `salida_cash_loans.txt`: salida completa de la consola.
