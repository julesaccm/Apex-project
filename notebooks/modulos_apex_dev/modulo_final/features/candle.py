"""
features/candle.py
==================
Features basadas en la morfología de la vela:

  · Ratios cuerpo / mechas / rango
  · Patrones de reversión           (Hammer, Shooting Star, Doji, Engulfing,
                                     Marubozu, Pinbar)
  · Rachas consecutivas             (velas rojas / verdes)
  · Distancias al High/Low de N velas
  · Posición del cierre dentro del rango de N velas
  · Retornos multi-período          (1, 3, 6, 12 velas)
"""

import numpy as np
import pandas as pd
from ._utils import safe_div, streak


class CandleFeatures:
    """
    Parámetros
    ----------
    range_windows : list  Ventanas para distancias y posición (default [10, 20, 50])
    return_lags   : list  Rezagos de retorno (en velas, default [1, 3, 6, 12])
    """

    def __init__(
        self,
        range_windows: list = None,
        return_lags  : list = None,
    ):
        self.range_windows = range_windows or [10, 20, 50]
        self.return_lags   = return_lags   or [1, 3, 6, 12]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        c, o, h, l = df["Close"], df["Open"], df["High"], df["Low"]
        out  = pd.DataFrame(index=df.index)
        rng  = (h - l).replace(0, np.nan)

        body       = (c - o).abs()
        body_top   = pd.concat([c, o], axis=1).max(axis=1)
        body_bot   = pd.concat([c, o], axis=1).min(axis=1)
        wick_upper = h - body_top
        wick_lower = body_bot - l

        # ── Ratios básicos ────────────────────────────────────────────────────
        out["Body_ratio"]          = safe_div(body,       rng)
        out["Wick_upper_ratio"]    = safe_div(wick_upper, rng)
        out["Wick_lower_ratio"]    = safe_div(wick_lower, rng)
        out["Is_bullish"]          = (c > o).astype(int)
        out["Candle_range_pct"]    = safe_div(rng, c) * 100

        # Mechas en valor absoluto (útiles como features de volatilidad)
        out["Wick_lower_abs_pct"]  = safe_div(wick_lower, c) * 100
        out["Wick_upper_abs_pct"]  = safe_div(wick_upper, c) * 100

        # ── Patrones de reversión ─────────────────────────────────────────────
        # Hammer (mínimo potencial): mecha inferior larga, cuerpo pequeño arriba
        out["Hammer"]              = (
            (out["Wick_lower_ratio"] > 0.55) &
            (out["Body_ratio"]       < 0.30) &
            (out["Is_bullish"]       == 1)
        ).astype(int)

        # Shooting Star (máximo potencial): mecha superior larga, cuerpo pequeño
        out["Shooting_star"]       = (
            (out["Wick_upper_ratio"] > 0.55) &
            (out["Body_ratio"]       < 0.30) &
            (out["Is_bullish"]       == 0)
        ).astype(int)

        # Doji: cuerpo casi nulo
        out["Doji"]                = (out["Body_ratio"] < 0.05).astype(int)

        # Dragonfly Doji (alcista): doji con mecha inferior larga
        out["Dragonfly_doji"]      = (
            out["Doji"] & (out["Wick_lower_ratio"] > 0.60)
        ).astype(int)

        # Gravestone Doji (bajista): doji con mecha superior larga
        out["Gravestone_doji"]     = (
            out["Doji"] & (out["Wick_upper_ratio"] > 0.60)
        ).astype(int)

        # Pinbar (rechazo fuerte, cualquier dirección)
        out["Pinbar"]              = (
            (out["Wick_lower_ratio"] > 0.65) | (out["Wick_upper_ratio"] > 0.65)
        ).astype(int)

        # Bullish Engulfing
        out["Bull_engulfing"]      = (
            (o > c.shift(1)) & (c > o.shift(1)) &
            (body > body.shift(1))  & (c.shift(1) < o.shift(1))
        ).astype(int)

        # Bearish Engulfing
        out["Bear_engulfing"]      = (
            (o < c.shift(1)) & (c < o.shift(1)) &
            (body > body.shift(1))  & (c.shift(1) > o.shift(1))
        ).astype(int)

        # Marubozu (fuerza pura sin mechas)
        out["Bull_marubozu"]       = (
            (out["Is_bullish"] == 1) & (out["Body_ratio"] > 0.90)
        ).astype(int)
        out["Bear_marubozu"]       = (
            (out["Is_bullish"] == 0) & (out["Body_ratio"] > 0.90)
        ).astype(int)

        # ── Rachas ────────────────────────────────────────────────────────────
        is_red   = (c < o)
        is_green = (c > o)
        out["Consecutive_red"]     = streak(is_red)
        out["Consecutive_green"]   = streak(is_green)
        out["Streak_exhaust_bear"] = (out["Consecutive_red"]   >= 5).astype(int)
        out["Streak_exhaust_bull"] = (out["Consecutive_green"] >= 5).astype(int)

        # ── Distancias a extremos recientes ───────────────────────────────────
        for n in self.range_windows:
            hi_n = h.rolling(n).max()
            lo_n = l.rolling(n).min()
            out[f"Dist_high_{n}_pct"] = safe_div(hi_n - c, c) * 100
            out[f"Dist_low_{n}_pct"]  = safe_div(c - lo_n, c) * 100
            out[f"Pos_in_range_{n}"]  = safe_div(c - lo_n, hi_n - lo_n)

        # ── Retornos multi-período ────────────────────────────────────────────
        for lag in self.return_lags:
            out[f"Return_{lag}p"]     = c.pct_change(lag)

        # ── Aceleración del precio (segunda derivada) ─────────────────────────
        out["Price_acceleration"]  = c.diff().diff()   # ∆∆close
        out["Return_momentum"]     = c.pct_change(3) - c.pct_change(6)  # short vs long

        return out
