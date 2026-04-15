"""
Detección de extremos locales en series OHLC — versión unificada
================================================================
Combina lo mejor de ambos enfoques:
  · High/Low separados para máximos/mínimos (precisión OHLC)
  · Savitzky-Golay como suavizador simétrico (sin lag)
  · Búsqueda en ventana ±n//2 para anclar al precio real
  · Reemplazo de extremo débil cuando aparece uno más extremo del mismo tipo
  · extremo_strength como magnitud normalizada del cambio
  · Distancia por posición entera (robusta ante huecos de calendario)
"""
 
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from typing import Literal
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
 
def _savgol_window(n: int, series_len: int, polyorder: int = 3) -> int:
    """Calcula un window_length válido para savgol_filter."""
    wl = n if n % 2 == 1 else n + 1          # debe ser impar
    wl = max(wl, polyorder + 2)               # debe superar el grado del polinomio
    wl = min(wl, series_len if series_len % 2 == 1 else series_len - 1)
    return wl
 
 
def _gradient(arr: np.ndarray) -> np.ndarray:
    """Gradiente centrado con bordes de primer orden."""
    d = np.empty_like(arr)
    d[0]  = arr[1] - arr[0]
    d[-1] = arr[-1] - arr[-2]
    d[1:-1] = (arr[2:] - arr[:-2]) / 2.0
    return d
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Función principal
# ─────────────────────────────────────────────────────────────────────────────
 
def etiquetar_extremos_validados(
    df: pd.DataFrame,
    ventana_critica: int   = 7,
    cambio_minimo: float   = 0.02,
    smooth_poly: int       = 3,
    col_high: str          = "High",
    col_low:  str          = "Low",
) -> pd.DataFrame:
    """
    Encuentra máximos y mínimos locales en datos OHLC con múltiples criterios.
 
    Parámetros
    ----------
    df               : DataFrame con columnas High y Low (DatetimeIndex).
    ventana_critica  : Ventana n. Controla:
                         - radio de búsqueda del extremo real (±n//2)
                         - separación mínima entre extremos consecutivos
                         - window_length del filtro Savitzky-Golay
    cambio_minimo    : Variación porcentual mínima entre extremos (default 0.02 = 2%).
    smooth_poly      : Orden del polinomio del filtro SG (default 3).
    col_high / col_low: Nombres de columnas de precio máximo/mínimo.
 
    Retorna
    -------
    pd.DataFrame con las columnas originales más:
        extremo_tipo      : 0=neutral | 1=máximo | -1=mínimo
        extremo_validado  : 0=candidato descartado | 1=validado
        extremo_strength  : intensidad normalizada [0,1] (|Δprecio| / precio_base)
        extremo_precio    : precio del extremo (High si max, Low si min)
        d1                : primera derivada suavizada en el punto
        d2                : segunda derivada suavizada en el punto
        pct_change        : cambio % respecto al extremo anterior validado
    """
    out = df.copy()
    n   = ventana_critica
    N   = len(out)
 
    # ── 1. Savitzky-Golay sobre High y Low ────────────────────────────────────
    wl = _savgol_window(n, N, smooth_poly)
 
    high_vals = out[col_high].values.astype(float)
    low_vals  = out[col_low].values.astype(float)
 
    high_sg = savgol_filter(high_vals, window_length=wl, polyorder=smooth_poly)
    low_sg  = savgol_filter(low_vals,  window_length=wl, polyorder=smooth_poly)
 
    # ── 2. Derivadas sobre series suavizadas ─────────────────────────────────
    d1_high = _gradient(high_sg)
    d2_high = _gradient(d1_high)
 
    d1_low  = _gradient(low_sg)
    d2_low  = _gradient(d1_low)
 
    # ── 3. Inicializar columnas de salida ────────────────────────────────────
    out["extremo_tipo"]     = 0
    out["extremo_validado"] = 0
    out["extremo_strength"] = 0.0
    out["extremo_precio"]   = np.nan
    out["d1"]               = 0.0
    out["d2"]               = 0.0
    out["pct_change"]       = np.nan
 
    # ── 4. Detectar candidatos por cruces de cero de d1 ──────────────────────
    # Máximos: d1 positiva → negativa  &  d2 < 0 (curvatura cóncava)
    # Mínimos: d1 negativa → positiva  &  d2 > 0 (curvatura convexa)
    candidatos: list[dict] = []
    half = n // 2
 
    for i in range(N - 1):
        cruce_max = (d1_high[i] > 0) and (d1_high[i + 1] <= 0) and (d2_high[i] < 0)
        cruce_min = (d1_low[i]  < 0) and (d1_low[i + 1]  >= 0) and (d2_low[i]  > 0)
 
        if not (cruce_max or cruce_min):
            continue
 
        etype: Literal[1, -1] = 1 if cruce_max else -1
 
        # ── Condición 4: buscar el precio real dentro de ±n//2 ───────────────
        lo = max(0, i - half)
        hi = min(N - 1, i + half)
 
        if etype == 1:        # máximo: busca en High
            local_off = int(np.argmax(high_vals[lo:hi + 1]))
            actual_idx = lo + local_off
            precio     = high_vals[actual_idx]
            d1_val     = d1_high[actual_idx]
            d2_val     = d2_high[actual_idx]
        else:                 # mínimo: busca en Low
            local_off = int(np.argmin(low_vals[lo:hi + 1]))
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
 
    # ── 5. Deduplicar: mismo índice → conservar el de mayor |d2| ─────────────
    seen: dict[int, dict] = {}
    for c in candidatos:
        if c["pos"] not in seen or abs(c["d2"]) > abs(seen[c["pos"]]["d2"]):
            seen[c["pos"]] = c
    candidatos = sorted(seen.values(), key=lambda x: x["pos"])
 
    # ── 6. Filtros de validación ─────────────────────────────────────────────
    validados: list[dict] = []
 
    for c in candidatos:
        if not validados:
            validados.append(c)
            continue
 
        last     = validados[-1]
        distancia = c["pos"] - last["pos"]   # entero posicional, robusto ante huecos
 
        # ── Condición 2: separación mínima de n periodos ─────────────────────
        if distancia < n:
            # Demasiado cerca → reemplazar si mismo tipo y más extremo.
            # Se hereda pct_change del reemplazado para no perder la referencia
            # al extremo anterior de tipo distinto.
            if c["tipo"] == last["tipo"]:
                if (c["tipo"] ==  1 and c["precio"] > last["precio"]) or \
                   (c["tipo"] == -1 and c["precio"] < last["precio"]):
                    c["pct_change"] = last.get("pct_change", np.nan)
                    validados[-1] = c
            # Tipos distintos muy cercanos → descartar el nuevo
            continue
 
        # ── Alternancia + reemplazo si mismo tipo con distancia ≥ n ──────────
        if c["tipo"] == last["tipo"]:
            if (c["tipo"] ==  1 and c["precio"] >= last["precio"]) or \
               (c["tipo"] == -1 and c["precio"] <= last["precio"]):
                # Heredar pct_change: el mejor candidato del mismo tipo
                # mantiene la distancia porcentual respecto al extremo opuesto previo.
                c["pct_change"] = last.get("pct_change", np.nan)
                validados[-1] = c
            continue
 
        # ── Condición 3: cambio porcentual mínimo ────────────────────────────
        pct = abs(c["precio"] - last["precio"]) / last["precio"]
        if pct < cambio_minimo:
            continue
 
        c["pct_change"] = pct
        validados.append(c)
 
    # ── 7. Calcular strength normalizado y escribir en DataFrame ─────────────
    # strength = |Δprecio respecto al extremo anterior| / precio_base → [0,1]
    for i, c in enumerate(validados):
        idx = c["pos"]
        pct = c.get("pct_change", np.nan)
 
        out.iloc[idx, out.columns.get_loc("extremo_tipo")]     = c["tipo"]
        out.iloc[idx, out.columns.get_loc("extremo_validado")] = 1
        out.iloc[idx, out.columns.get_loc("extremo_precio")]   = c["precio"]
        out.iloc[idx, out.columns.get_loc("d1")]               = round(c["d1"], 6)
        out.iloc[idx, out.columns.get_loc("d2")]               = round(c["d2"], 6)
        out.iloc[idx, out.columns.get_loc("pct_change")]       = round(pct, 4) if not np.isnan(pct) else np.nan
        # strength: pct ya es fracción, clip a [0,1] por seguridad
        out.iloc[idx, out.columns.get_loc("extremo_strength")] = round(min(float(pct) if not np.isnan(pct) else 0.0, 1.0), 4)
 
    return out