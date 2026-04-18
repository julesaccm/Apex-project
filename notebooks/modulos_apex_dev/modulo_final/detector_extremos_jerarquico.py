"""
etiquetado_jerarquico.py
========================
Etiquetado jerárquico de extremos en dos niveles de confianza:

  Clase 2 — Alta confianza   : extremos validados por Savitzky-Golay (detector_extremos.py)
  Clase 1 — Baja confianza   : extremos detectados únicamente por Trend Scanning (López de Prado)
  Clase 0 — Neutral          : el resto de observaciones

Motivación
----------
SG aplica filtros estrictos (alternancia, separación mínima, Δ%≥2%).
Eso produce etiquetas limpias pero pocas (~10-20 por año en datos diarios).
Trend Scanning detecta cambios de régimen estadísticamente significativos
sin imponer esas restricciones, añadiendo muestras reales que SG rechazó por
criterios geométricos, no por ser ruido.

La combinación se usa con sample_weight durante el entrenamiento:
  · Clase 2 → peso 1.0 (señal fuerte)
  · Clase 1 → peso configurable (default 0.4)
  · Clase 0 → peso 0.0 (excluido del entrenamiento de extremos)

Referencia Trend Scanning
--------------------------
López de Prado, M. (2020). Machine Learning for Asset Managers.
Cambridge University Press. Sección 5.4.

Algoritmo implementado
----------------------
Para cada punto t, se ajustan regresiones OLS forward-looking en horizontes
L ∈ [min_L, max_L]. Se selecciona el L que maximiza |t-statistic| del
coeficiente de pendiente. El signo del t-stat da la dirección del trend.
Una transición de régimen (cambio de signo de bin) + |t_val| > umbral
define un candidato a extremo de baja confianza.

Uso
---
    from etiquetado_jerarquico import etiquetar_jerarquico

    resultado = etiquetar_jerarquico(df, ventana_critica=14)
    # Columnas nuevas: label_hier, label_conf, ts_t_val, sample_weight
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from .detector_extremos import etiquetar_extremos_validados  # módulo del proyecto


# ─────────────────────────────────────────────────────────────────────────────
# 1. Trend Scanning (vectorizado, sin dependencias externas)
# ─────────────────────────────────────────────────────────────────────────────

def _trend_scanning(
    close   : pd.Series,
    min_L   : int   = 3,
    max_L   : int   = 21,
) -> pd.DataFrame:
    """
    Trend Scanning de López de Prado sobre una serie de precios de cierre.

    Para cada t ajusta OLS en la ventana forward [t, t+L] para cada L
    ∈ [min_L, max_L] y selecciona el horizonte con mayor |t-statistic|
    del coeficiente de pendiente.

    Implementación OLS manual (sin scipy.stats.linregress) para
    evitar overhead de función Python en el bucle interior.

    Parámetros
    ----------
    close : serie de precios de cierre con DatetimeIndex.
    min_L : horizonte mínimo de regresión (en periodos).
    max_L : horizonte máximo de regresión (en periodos).

    Retorna
    -------
    DataFrame con columnas:
        t_val  : t-statistic del slope más significativo en t
        L_opt  : horizonte óptimo seleccionado
        bin    : signo(t_val) → {-1, 0, 1}
    """
    vals  = close.values.astype(np.float64)
    N     = len(vals)
    t_arr = np.zeros(N, dtype=np.float64)
    L_arr = np.zeros(N, dtype=np.int32)

    for t in range(N):
        best_t = 0.0
        best_L = min_L
        for L in range(min_L, max_L + 1):
            end = t + L
            if end >= N:
                break
            y   = vals[t : end + 1]
            n   = len(y)
            x   = np.arange(n, dtype=np.float64)

            # OLS: β = Sxy/Sxx
            mx  = x.mean()
            my  = y.mean()
            Sxx = ((x - mx) ** 2).sum()
            if Sxx < 1e-10:
                continue

            b    = ((x - mx) * (y - my)).sum() / Sxx
            a    = my - b * mx
            rss  = ((y - (a + b * x)) ** 2).sum()
            se   = np.sqrt(rss / max(n - 2, 1) / Sxx)
            tval = b / se if se > 1e-12 else 0.0

            if abs(tval) > abs(best_t):
                best_t = tval
                best_L = L

        t_arr[t] = best_t
        L_arr[t] = best_L

    return pd.DataFrame(
        {"t_val": t_arr, "L_opt": L_arr, "bin": np.sign(t_arr).astype(int)},
        index=close.index,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Extracción de extremos desde resultado de Trend Scanning
# ─────────────────────────────────────────────────────────────────────────────

def _extremos_desde_ts(
    ts_df        : pd.DataFrame,
    close        : pd.Series,
    min_tval     : float = 2.0,
    embargo_dias : int   = 5,
) -> tuple[pd.Series, pd.Series]:
    """
    Convierte transiciones de régimen del Trend Scanning en etiquetas de extremos.

    Lógica de transición
    --------------------
      bin: -1 → +1  →  mínimo local  (mercado girando de bajista a alcista)
      bin: +1 → -1  →  máximo local  (mercado girando de alcista a bajista)

    Para cada transición detectada:
      1. Se filtra por |t_val| > min_tval (significancia estadística).
      2. Se busca el precio más extremo en una ventana ±embargo_dias alrededor
         de la transición (ancla al precio real, no al punto matemático).

    Parámetros
    ----------
    ts_df        : resultado de _trend_scanning().
    close        : serie de precios de cierre.
    min_tval     : umbral mínimo de |t-statistic| para aceptar la transición.
    embargo_dias : semi-ventana de búsqueda del precio real (en periodos).

    Retorna
    -------
    extremos : Series con 1=máximo, -1=mínimo, 0=neutral.
    tval_out : Series con el |t_val| en el punto del extremo (NaN si neutral).
    """
    N       = len(close)
    vals    = close.values
    bin_s   = ts_df["bin"]
    tval_s  = ts_df["t_val"]

    extremos = pd.Series(0,   index=close.index, dtype=int)
    tval_out = pd.Series(np.nan, index=close.index)

    for i in range(1, N):
        prev_bin = int(bin_s.iloc[i - 1])
        curr_bin = int(bin_s.iloc[i])

        # Sin cambio de régimen o régimen indeterminado
        if prev_bin == curr_bin or curr_bin == 0:
            continue

        # Filtro de significancia
        if abs(tval_s.iloc[i]) < min_tval:
            continue

        # Dirección del extremo
        etype = -1 if curr_bin == 1 else 1   # transición ↑ = mínimo, ↓ = máximo

        # Anclar al precio real dentro de la ventana de embargo
        lo = max(0, i - embargo_dias)
        hi = min(N - 1, i + embargo_dias)

        if etype == 1:
            best_pos = lo + int(np.argmax(vals[lo : hi + 1]))
        else:
            best_pos = lo + int(np.argmin(vals[lo : hi + 1]))

        extremos.iloc[best_pos] = etype
        tval_out.iloc[best_pos] = abs(tval_s.iloc[i])

    return extremos, tval_out


# ─────────────────────────────────────────────────────────────────────────────
# 3. Función principal de etiquetado jerárquico
# ─────────────────────────────────────────────────────────────────────────────

def etiquetar_jerarquico(
    df             : pd.DataFrame,
    ventana_critica: int   = 5,
    cambio_minimo  : float = 0.06,
    smooth_poly    : int   = 3,
    ts_min_L       : int   = 3,
    ts_max_L       : int   | None = None,
    ts_min_tval    : float = 2.0,
    ts_embargo     : int   | None = None,
    peso_baja      : float = 0.4,
    col_high       : str   = "High",
    col_low        : str   = "Low",
    col_close      : str   = "Close",

) -> pd.DataFrame:
    """
    Etiquetado jerárquico de extremos en dos niveles de confianza.

    Parámetros
    ----------
    df              : DataFrame OHLC con DatetimeIndex.
    ventana_critica : Ventana n para el detector SG (también usada como max_L
                      del Trend Scanning si ts_max_L=None).
    cambio_minimo   : Δ% mínimo entre extremos SG (default 2%).
    smooth_poly     : Grado del polinomio SG (default 3).
    ts_min_L        : Horizonte mínimo del Trend Scanning.
    ts_max_L        : Horizonte máximo del Trend Scanning.
                      Si None → usa ventana_critica.
    ts_min_tval     : Umbral mínimo |t-stat| para aceptar extremo TS.
    ts_embargo      : Semi-ventana para anclar el precio real en TS.
                      Si None → n//2.
    peso_baja       : sample_weight asignado a extremos de clase 1 (baja conf.).
    col_high/low/close: Nombres de columnas OHLC.

    Retorna
    -------
    DataFrame con columnas originales + nuevas columnas:
        label_hier    : 0=neutral | 1=baja confianza (TS) | 2=alta confianza (SG)
        label_tipo    : 0=neutral | 1=máximo | -1=mínimo
        label_conf    : 'neutral' | 'ts_low' | 'sg_high'
        ts_t_val      : |t_val| del Trend Scanning en el punto (NaN si no es TS)
        sample_weight : peso de entrenamiento (0=neutral, peso_baja=TS, 1.0=SG)

        + todas las columnas del detector SG:
          extremo_tipo, extremo_validado, extremo_strength,
          extremo_precio, d1, d2, pct_change
    """
    n          = ventana_critica
    max_L      = ts_max_L   if ts_max_L   is not None else n
    embargo    = ts_embargo if ts_embargo is not None else n // 2

    # ── Paso 1: SG — extremos de alta confianza ───────────────────────────────
    res_sg = etiquetar_extremos_validados(
        df,
        ventana_critica = n,
        cambio_minimo   = cambio_minimo,
        smooth_poly     = smooth_poly,
        col_high        = col_high,
        col_low         = col_low
    )
    sg_tipo = res_sg["extremo_tipo"]   # -1, 0, 1

    # ── Paso 2: Trend Scanning — detección de cambios de régimen ─────────────
    close   = df[col_close]
    ts_df   = _trend_scanning(close, min_L=ts_min_L, max_L=max_L)
    ts_ext, ts_tval = _extremos_desde_ts(
        ts_df,
        close,
        min_tval     = ts_min_tval,
        embargo_dias = embargo,
    )

    # ── Paso 3: Fusión jerárquica ─────────────────────────────────────────────
    # Reglas de precedencia:
    #   · Si SG marca un extremo → clase 2 (alta confianza), ignorar TS en ese punto.
    #   · Si TS marca un extremo Y SG no → clase 1 (baja confianza).
    #   · Conflicto (mismo punto, tipos distintos) → SG gana.
    #   · Neutral → clase 0.

    label_hier = pd.Series(0,    index=df.index, dtype=int)
    label_tipo = pd.Series(0,    index=df.index, dtype=int)
    label_conf = pd.Series("neutral", index=df.index, dtype=object)
    ts_tval_col = pd.Series(np.nan, index=df.index)

    # Clase 1: solo en TS
    mask_ts_solo = (ts_ext != 0) & (sg_tipo == 0)
    label_hier[mask_ts_solo]    = 1
    label_tipo[mask_ts_solo]    = ts_ext[mask_ts_solo]
    label_conf[mask_ts_solo]    = "ts_low"
    ts_tval_col[mask_ts_solo]   = ts_tval[mask_ts_solo]

    # Clase 2: SG (sobreescribe cualquier TS en el mismo punto)
    mask_sg = sg_tipo != 0
    label_hier[mask_sg]  = 2
    label_tipo[mask_sg]  = sg_tipo[mask_sg]
    label_conf[mask_sg]  = "sg_high"
    ts_tval_col[mask_sg] = np.nan   # TS irrelevante donde SG ya es autoritativo

    # ── Paso 4: sample_weight ─────────────────────────────────────────────────
    weight_map   = {0: 0.0, 1: peso_baja, 2: 1.0}
    sample_weight = label_hier.map(weight_map)

    # ── Paso 5: Ensamblar DataFrame final ─────────────────────────────────────
    out = res_sg.copy()
    out["label_hier"]    = label_hier
    out["label_tipo"]    = label_tipo
    out["label_conf"]    = label_conf
    out["ts_t_val"]      = ts_tval_col.round(4)
    out["sample_weight"] = sample_weight

    # Adjuntar el Trend Scanning completo como atributo (útil para análisis)
    out.attrs["ts_full"] = ts_df

    return out


# ─────────────────────────────────────────────────────────────────────────────
# 4. Utilidades de análisis
# ─────────────────────────────────────────────────────────────────────────────

def resumen_etiquetas(resultado: pd.DataFrame) -> pd.DataFrame:
    """
    Devuelve una tabla de resumen con estadísticas de las etiquetas generadas.
    """
    out = resultado
    hier = out["label_hier"]
    tipo = out["label_tipo"]
    sw   = out["sample_weight"]

    sg_max  = ((hier == 2) & (tipo ==  1)).sum()
    sg_min  = ((hier == 2) & (tipo == -1)).sum()
    ts_max  = ((hier == 1) & (tipo ==  1)).sum()
    ts_min  = ((hier == 1) & (tipo == -1)).sum()

    sg_pct_changes = out.loc[(hier == 2), "pct_change"].dropna()
    ts_tvals       = out.loc[(hier == 1), "ts_t_val"].dropna()

    filas = [
        {"clase": "2 — SG alta conf.",  "máximos": sg_max, "mínimos": sg_min,
         "total": sg_max + sg_min,
         "peso_total": sw[hier == 2].sum().round(1),
         "Δ%_medio": f"{sg_pct_changes.mean()*100:.1f}%" if len(sg_pct_changes) else "—",
         "t_val_medio": "—"},
        {"clase": "1 — TS baja conf.",  "máximos": ts_max, "mínimos": ts_min,
         "total": ts_max + ts_min,
         "peso_total": sw[hier == 1].sum().round(1),
         "Δ%_medio": "—",
         "t_val_medio": f"{ts_tvals.mean():.2f}" if len(ts_tvals) else "—"},
        {"clase": "0 — Neutral",        "máximos": 0, "mínimos": 0,
         "total": (hier == 0).sum(),
         "peso_total": 0.0, "Δ%_medio": "—", "t_val_medio": "—"},
    ]
    return pd.DataFrame(filas).set_index("clase")


def split_con_purga(
    resultado   : pd.DataFrame,
    n           : int,
    test_ratio  : float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    División temporal train/test con purga de n//2 filas al final del train.

    La purga elimina las filas cuyas etiquetas SG usan datos del periodo de test
    (efecto del suavizador simétrico Savitzky-Golay).

    Retorna (train_df, test_df) sin el bloque de purga.
    """
    half  = n // 2
    N     = len(resultado)
    cut   = int(N * (1 - test_ratio))

    train = resultado.iloc[: cut - half]
    test  = resultado.iloc[cut :]
    return train, test


def walk_forward_folds(
    resultado  : pd.DataFrame,
    n          : int,
    n_folds    : int = 5,
    min_train  : int = 252,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Generador de folds para walk-forward validation con purga en cada split.

    Retorna lista de (train_df, test_df).
    """
    half      = n // 2
    N         = len(resultado)
    available = N - min_train
    fold_size = available // n_folds

    folds = []
    for k in range(n_folds):
        test_start = min_train + k * fold_size
        test_end   = test_start + fold_size if k < n_folds - 1 else N
        train_end  = test_start - half

        if train_end <= 0 or test_start >= N:
            continue

        train = resultado.iloc[:train_end]
        test  = resultado.iloc[test_start:test_end]
        folds.append((train, test))

    return folds


# # ─────────────────────────────────────────────────────────────────────────────
# # 5. Demo
# # ─────────────────────────────────────────────────────────────────────────────

# def _generar_ohlc(n: int = 800, seed: int = 42) -> pd.DataFrame:
#     rng   = np.random.default_rng(seed)
#     dates = pd.date_range("2017-01-01", periods=n, freq="D")
#     t     = np.linspace(0, 8 * np.pi, n)
#     close = (
#         30_000 + 20_000 * (np.arange(n) / n)
#         + 8_000 * np.sin(t)
#         + 3_000 * np.sin(2.3 * t)
#         + 1_200 * np.sin(5 * t)
#         + rng.normal(0, 800, n).cumsum() * 0.18
#     ).clip(min=10_000)
#     sp = rng.uniform(300, 1_200, n)
#     return pd.DataFrame(
#         {"Open": close + rng.normal(0, 200, n),
#          "High": close + sp,
#          "Low" : close - sp,
#          "Close": close},
#         index=dates,
#     )


# if __name__ == "__main__":
#     import time

#     df = _generar_ohlc(800)
#     n  = 14

#     print("Ejecutando etiquetado jerárquico…")
#     t0  = time.time()
#     res = etiquetar_jerarquico(df, ventana_critica=n, ts_min_tval=2.0, peso_baja=0.4)
#     print(f"Completado en {time.time()-t0:.2f}s\n")

#     # Resumen
#     print(resumen_etiquetas(res).to_string())

#     # Distribución de sample_weight
#     sw = res["sample_weight"]
#     print(f"\nSample weight — suma efectiva positivos: {sw[sw>0].sum():.1f}")
#     print(f"Equivalente SG puro sería: {(res['label_hier']==2).sum():.0f}")
#     print(f"Ganancia ponderada: +{(sw[sw>0].sum() - (res['label_hier']==2).sum()):.1f} unidades")

#     # Verificar purga
#     train, test = split_con_purga(res, n=n, test_ratio=0.20)
#     half = n // 2
#     max_train = train.index.max()
#     min_test  = test.index.min()
#     horizonte = max_train + pd.Timedelta(days=half)
#     ok = horizonte < min_test
#     print(f"\nPurga: último train={max_train.date()}  "
#           f"horizonte SG={horizonte.date()}  "
#           f"primer test={min_test.date()}  "
#           f"{'✅ sin leakage' if ok else '❌ leakage!'}")

#     # Walk-forward
#     folds = walk_forward_folds(res, n=n, n_folds=4, min_train=200)
#     print(f"\nWalk-forward {len(folds)} folds:")
#     for i, (tr, te) in enumerate(folds):
#         sg_tr  = (tr["label_hier"] == 2).sum()
#         ts_tr  = (tr["label_hier"] == 1).sum()
#         sg_te  = (te["label_hier"] == 2).sum()
#         gap    = (te.index.min() - tr.index.max()).days
#         print(f"  Fold {i+1}: train={len(tr)} (SG={sg_tr} TS={ts_tr})  "
#               f"test={len(te)} (SG={sg_te})  gap_purga={gap}d")

#     # Ejemplo de uso con XGBoost
#     print("\n── Ejemplo entrenamiento binario (máximos, clase SG+TS) ────────────")
#     from sklearn.preprocessing import StandardScaler
#     try:
#         from xgboost import XGBClassifier
#         from sklearn.metrics import precision_score, recall_score

#         # Features simples (solo para demo)
#         feat = pd.DataFrame({
#             "ret5"  : df["Close"].pct_change(5),
#             "ret14" : df["Close"].pct_change(14),
#             "vol14" : df["Close"].pct_change().rolling(14).std(),
#             "sma_r" : df["Close"].rolling(7).mean() / df["Close"] - 1,
#         }).dropna()

#         common = res.index.intersection(feat.index)
#         res_c  = res.loc[common]
#         feat_c = feat.loc[common]

#         train, test = split_con_purga(res_c, n=n)
#         Xtr = feat_c.loc[train.index].dropna()
#         Xte = feat_c.loc[test.index].dropna()
#         train_a = train.loc[Xtr.index]
#         test_a  = test.loc[Xte.index]

#         # Target: ¿es este punto un máximo (de cualquier nivel)?
#         ytr = (train_a["label_tipo"] == 1).astype(int)
#         yte = (test_a["label_tipo"]  == 1).astype(int)
#         wtr = train_a["sample_weight"].values

#         pos_ratio = ytr.sum() / len(ytr)
#         spw = (1 - pos_ratio) / pos_ratio if pos_ratio > 0 else 1

#         model = XGBClassifier(n_estimators=200, max_depth=4,
#                                learning_rate=0.05, scale_pos_weight=spw,
#                                eval_metric="logloss", random_state=42,
#                                verbosity=0)

#         # Entrenamiento sin peso (solo SG)
#         ytr_sg = (train_a["extremo_tipo"] == 1).astype(int)
#         model_sg = XGBClassifier(n_estimators=200, max_depth=4,
#                                   learning_rate=0.05, scale_pos_weight=spw,
#                                   eval_metric="logloss", random_state=42,
#                                   verbosity=0)
#         model_sg.fit(Xtr, ytr_sg)
#         p_sg = precision_score(yte, model_sg.predict(Xte), zero_division=0)
#         r_sg = recall_score(yte,    model_sg.predict(Xte), zero_division=0)

#         # Entrenamiento con sample_weight (SG + TS ponderado)
#         model.fit(Xtr, ytr, sample_weight=wtr)
#         p_j = precision_score(yte, model.predict(Xte), zero_division=0)
#         r_j = recall_score(yte,    model.predict(Xte), zero_division=0)

#         print(f"  Solo SG:      Precision={p_sg:.3f}  Recall={r_sg:.3f}")
#         print(f"  SG + TS (w):  Precision={p_j:.3f}  Recall={r_j:.3f}")

#     except ImportError:
#         print("  (xgboost no disponible — instalar con: pip install xgboost)")