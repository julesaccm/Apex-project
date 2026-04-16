"""
features/extreme_dist.py
========================
Variables de DISTANCIA Y TIEMPO desde el último extremo local.
Estas features son especialmente predictivas: cuando el precio lleva
muchas velas sin hacer un nuevo máximo/mínimo y se acerca al nivel
anterior, la probabilidad de reversión aumenta.

Features generadas
------------------
  Bars_since_last_max    : velas transcurridas desde el último máximo local
  Bars_since_last_min    : velas transcurridas desde el último mínimo local
  Pct_from_last_max      : % de caída desde el último máximo local (drawdown)
  Pct_from_last_min      : % de subida desde el último mínimo local (rally)
  Max_drawdown_since_max : mayor caída % intra-tramo desde el último máximo
  Max_rally_since_min    : mayor subida % intra-tramo desde el último mínimo
  Extreme_ratio          : ratio barras_desde_max / barras_desde_min
  Is_near_last_max       : bandera: precio a < near_threshold% del último máximo
  Is_near_last_min       : bandera: precio a < near_threshold% del último mínimo

Nota: requiere target_max y target_min calculados por Labeler.
      Si no están disponibles, usa umbrales de ventana rodante.
"""

import numpy as np
import pandas as pd
from ._utils import safe_div


class ExtremeDistanceFeatures:
    """
    Parámetros
    ----------
    ventana_critica : int    Ventana para detectar extremos locales si no hay targets
    near_threshold  : float  % para considerar que el precio "está cerca" del extremo
    """

    def __init__(self, ventana_critica: int = 5, near_threshold: float = 0.005):
        self.ventana   = ventana_critica
        self.near_thr  = near_threshold

    # ── Detecta extremos locales si no hay columna target ────────────────────
    def _detect_extremes(self, df: pd.DataFrame):
        w   = self.ventana * 2 + 1
        max_loc = df["High"].rolling(window=w, center=True).max()
        min_loc = df["Low"].rolling(window=w, center=True).min()
        is_max  = (df["High"] == max_loc)
        is_min  = (df["Low"]  == min_loc)
        return is_max, is_min

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        c   = df["Close"]
        out = pd.DataFrame(index=df.index)

        # ── Obtener señales de extremo ────────────────────────────────────────
        if "target_max" in df.columns and "target_min" in df.columns:
            is_max = df["target_max"].astype(bool)
            is_min = df["target_min"].astype(bool)
        else:
            is_max, is_min = self._detect_extremes(df)

        # ── Precio del último extremo y barras desde él ───────────────────────
        price_at_max = df["High"].where(is_max)
        price_at_min = df["Low"].where(is_min)

        last_max_price = price_at_max.ffill()
        last_min_price = price_at_min.ffill()

        # Conteo de barras desde el último extremo (reset en cada extremo)
        bars_since_max = pd.Series(np.nan, index=df.index)
        bars_since_min = pd.Series(np.nan, index=df.index)
        cnt_max = cnt_min = 0

        for i in range(len(df)):
            cnt_max += 1
            cnt_min += 1
            if is_max.iloc[i]:
                cnt_max = 0
            if is_min.iloc[i]:
                cnt_min = 0
            bars_since_max.iloc[i] = cnt_max
            bars_since_min.iloc[i] = cnt_min

        out["Bars_since_last_max"]    = bars_since_max
        out["Bars_since_last_min"]    = bars_since_min

        # ── Distancia porcentual al último extremo ────────────────────────────
        out["Pct_from_last_max"]      = safe_div(last_max_price - c, last_max_price) * 100
        out["Pct_from_last_min"]      = safe_div(c - last_min_price, last_min_price) * 100

        # ── Drawdown máximo intra-tramo desde el último máximo ────────────────
        # (la mayor caída % que hubo desde ese máximo hasta ahora)
        running_drawdown = pd.Series(0.0, index=df.index)
        running_rally    = pd.Series(0.0, index=df.index)
        peak_val = trough_val = float("nan")

        for i in range(len(df)):
            if is_max.iloc[i] or np.isnan(peak_val):
                peak_val = df["High"].iloc[i]
            if is_min.iloc[i] or np.isnan(trough_val):
                trough_val = df["Low"].iloc[i]

            ci = c.iloc[i]
            if not np.isnan(peak_val) and peak_val > 0:
                running_drawdown.iloc[i] = (peak_val - ci) / peak_val * 100
            if not np.isnan(trough_val) and trough_val > 0:
                running_rally.iloc[i]    = (ci - trough_val) / trough_val * 100

        out["Drawdown_since_last_max"]= running_drawdown
        out["Rally_since_last_min"]   = running_rally

        # ── Ratio barras (¿cuánto más tiempo desde max que desde min?) ────────
        out["Extreme_bar_ratio"]      = safe_div(bars_since_max, bars_since_min + 1)

        # ── Banderas: ¿está cerca del último extremo? ─────────────────────────
        out["Is_near_last_max"]       = (
            safe_div((last_max_price - c).abs(), c) < self.near_thr
        ).astype(int)
        out["Is_near_last_min"]       = (
            safe_div((last_min_price - c).abs(), c) < self.near_thr
        ).astype(int)

        # ── Velocidad de recuperación ─────────────────────────────────────────
        # % de movimiento por barra desde el último extremo
        out["Speed_from_last_max"]    = safe_div(
            out["Pct_from_last_max"], bars_since_max.replace(0, 1)
        )
        out["Speed_from_last_min"]    = safe_div(
            out["Pct_from_last_min"], bars_since_min.replace(0, 1)
        )

        return out
