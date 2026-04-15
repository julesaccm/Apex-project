"""
detector_extremos.py
====================
Detección de extremos locales en series OHLC con Savitzky-Golay.

Algoritmo original del usuario, empaquetado como módulo del proyecto.

Lógica
------
  1. Suaviza High y Low con filtro Savitzky-Golay (sin lag, simétrico).
  2. Calcula primera (d1) y segunda (d2) derivada sobre las series suavizadas.
  3. Detecta candidatos por cruce de cero de d1 con curvatura correcta (d2).
  4. Ancla cada candidato al High/Low real dentro de ±n//2 velas.
  5. Deduplica (mismo índice → mayor |d2|).
  6. Valida con tres filtros:
       · Separación mínima de n velas entre extremos consecutivos.
       · Alternancia obligatoria (max → min → max → …), con reemplazo
         si aparece uno más extremo del mismo tipo.
       · Cambio porcentual mínimo entre extremos consecutivos (cambio_minimo).
  7. Calcula extremo_strength = |Δprecio| / precio_base, normalizado [0, 1].

Columnas añadidas al DataFrame
-------------------------------
  extremo_tipo      : 0=neutral | 1=máximo | -1=mínimo
  extremo_validado  : 1=validado, 0=descartado/neutral
  extremo_strength  : intensidad [0, 1]
  extremo_precio    : High si máximo, Low si mínimo
  d1                : primera derivada suavizada en el punto
  d2                : segunda derivada suavizada en el punto
  pct_change        : cambio % respecto al extremo anterior validado
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from typing import Literal


# ─────────────────────────────────────────────────────────────────────────────
# Helpers internos
# ─────────────────────────────────────────────────────────────────────────────

def _savgol_window(n: int, series_len: int, polyorder: int = 3) -> int:
    """Calcula un window_length válido para savgol_filter."""
    wl = n if n % 2 == 1 else n + 1
    wl = max(wl, polyorder + 2)
    wl = min(wl, series_len if series_len % 2 == 1 else series_len - 1)
    return wl


def _gradient(arr: np.ndarray) -> np.ndarray:
    """Gradiente centrado con bordes de primer orden."""
    d = np.empty_like(arr)
    d[0]    = arr[1] - arr[0]
    d[-1]   = arr[-1] - arr[-2]
    d[1:-1] = (arr[2:] - arr[:-2]) / 2.0
    return d


# ─────────────────────────────────────────────────────────────────────────────
# Función principal
# ─────────────────────────────────────────────────────────────────────────────

def etiquetar_extremos_validados(
    df: pd.DataFrame,
    ventana_critica : int   = 7,
    cambio_minimo   : float = 0.02,
    smooth_poly     : int   = 3,
    col_high        : str   = "High",
    col_low         : str   = "Low",
) -> pd.DataFrame:
    """
    Encuentra máximos y mínimos locales en datos OHLC con múltiples criterios.

    Parámetros
    ----------
    df               : DataFrame con columnas High y Low (DatetimeIndex).
    ventana_critica  : Ventana n. Controla el radio de búsqueda (±n//2),
                       la separación mínima entre extremos y el window SG.
    cambio_minimo    : Variación porcentual mínima entre extremos (default 0.02 = 2 %).
    smooth_poly      : Orden del polinomio del filtro SG (default 3).
    col_high / col_low: Nombres de columnas de precio máximo / mínimo.

    Retorna
    -------
    pd.DataFrame con columnas originales + metadatos de extremos.
    """
    out = df.copy()
    n   = ventana_critica
    N   = len(out)

    # ── 1. Savitzky-Golay sobre High y Low ───────────────────────────────────
    wl = _savgol_window(n, N, smooth_poly)

    high_vals = out[col_high].values.astype(float)
    low_vals  = out[col_low].values.astype(float)

    high_sg = savgol_filter(high_vals, window_length=wl, polyorder=smooth_poly)
    low_sg  = savgol_filter(low_vals,  window_length=wl, polyorder=smooth_poly)

    # ── 2. Derivadas ─────────────────────────────────────────────────────────
    d1_high = _gradient(high_sg)
    d2_high = _gradient(d1_high)
    d1_low  = _gradient(low_sg)
    d2_low  = _gradient(d1_low)

    # ── 3. Inicializar columnas de salida ─────────────────────────────────────
    out["extremo_tipo"]     = 0
    out["extremo_validado"] = 0
    out["extremo_strength"] = 0.0
    out["extremo_precio"]   = np.nan
    out["d1"]               = 0.0
    out["d2"]               = 0.0
    out["pct_change"]       = np.nan

    # ── 4. Detectar candidatos por cruces de cero de d1 ──────────────────────
    candidatos: list[dict] = []
    half = n // 2

    for i in range(N - 1):
        cruce_max = (d1_high[i] > 0) and (d1_high[i + 1] <= 0) and (d2_high[i] < 0)
        cruce_min = (d1_low[i]  < 0) and (d1_low[i + 1]  >= 0) and (d2_low[i]  > 0)

        if not (cruce_max or cruce_min):
            continue

        etype: Literal[1, -1] = 1 if cruce_max else -1
        lo = max(0, i - half)
        hi = min(N - 1, i + half)

        if etype == 1:
            local_off  = int(np.argmax(high_vals[lo:hi + 1]))
            actual_idx = lo + local_off
            precio     = high_vals[actual_idx]
            d1_val     = d1_high[actual_idx]
            d2_val     = d2_high[actual_idx]
        else:
            local_off  = int(np.argmin(low_vals[lo:hi + 1]))
            actual_idx = lo + local_off
            precio     = low_vals[actual_idx]
            d1_val     = d1_low[actual_idx]
            d2_val     = d2_low[actual_idx]

        candidatos.append(
            dict(pos=actual_idx, tipo=etype, precio=precio,
                 d1=d1_val, d2=d2_val)
        )

    if not candidatos:
        return out

    # ── 5. Deduplicar: mismo índice → conservar mayor |d2| ───────────────────
    seen: dict[int, dict] = {}
    for c in candidatos:
        if c["pos"] not in seen or abs(c["d2"]) > abs(seen[c["pos"]]["d2"]):
            seen[c["pos"]] = c
    candidatos = sorted(seen.values(), key=lambda x: x["pos"])

    # ── 6. Filtros de validación ──────────────────────────────────────────────
    validados: list[dict] = []

    for c in candidatos:
        if not validados:
            validados.append(c)
            continue

        last      = validados[-1]
        distancia = c["pos"] - last["pos"]

        # Separación mínima de n periodos
        if distancia < n:
            if c["tipo"] == last["tipo"]:
                if (c["tipo"] ==  1 and c["precio"] > last["precio"]) or \
                   (c["tipo"] == -1 and c["precio"] < last["precio"]):
                    c["pct_change"] = last.get("pct_change", np.nan)
                    validados[-1]   = c
            continue

        # Alternancia + reemplazo si mismo tipo con distancia ≥ n
        if c["tipo"] == last["tipo"]:
            if (c["tipo"] ==  1 and c["precio"] >= last["precio"]) or \
               (c["tipo"] == -1 and c["precio"] <= last["precio"]):
                c["pct_change"] = last.get("pct_change", np.nan)
                validados[-1]   = c
            continue

        # Cambio porcentual mínimo
        pct = abs(c["precio"] - last["precio"]) / last["precio"]
        if pct < cambio_minimo:
            continue

        c["pct_change"] = pct
        validados.append(c)

    # ── 7. Escribir resultados en el DataFrame ────────────────────────────────
    for c in validados:
        idx = c["pos"]
        pct = c.get("pct_change", np.nan)
        strength = round(min(float(pct) if not np.isnan(pct) else 0.0, 1.0), 4)

        out.iloc[idx, out.columns.get_loc("extremo_tipo")]     = c["tipo"]
        out.iloc[idx, out.columns.get_loc("extremo_validado")] = 1
        out.iloc[idx, out.columns.get_loc("extremo_precio")]   = c["precio"]
        out.iloc[idx, out.columns.get_loc("d1")]               = round(c["d1"], 6)
        out.iloc[idx, out.columns.get_loc("d2")]               = round(c["d2"], 6)
        out.iloc[idx, out.columns.get_loc("pct_change")]       = round(pct, 4) if not np.isnan(pct) else np.nan
        out.iloc[idx, out.columns.get_loc("extremo_strength")] = strength

    return out
