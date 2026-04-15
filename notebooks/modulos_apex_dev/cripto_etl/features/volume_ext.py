"""
features/volume_ext.py
======================
Features de volumen NO cubiertas por technical.py (que ya tiene Volumen_Relativo):

  · VWAP diario                      y distancia al precio
  · OBV                              y su divergencia con precio
  · Volume Delta                     (proxy buy/sell con OHLC)
  · CVD (Cumulative Volume Delta)    y su divergencia
  · Volumen relativo multi-ventana   (10, 20, 50 velas)
  · Spike y sequía de volumen        (flags binarios)
  · Bid/Ask imbalance                (proxy o real si hay columnas)
  · Volumen en rachas alcistas/bajistas
"""

import numpy as np
import pandas as pd
from ._utils import safe_div, divergence


class VolumeExtFeatures:
    """
    Parámetros
    ----------
    vol_windows   : list  Ventanas para volumen relativo (default [10, 20, 50])
    spike_mult    : float Multiplicador sobre la media para calificar spike (default 2.5)
    dry_mult      : float Multiplicador (por debajo) para calificar sequía (default 0.4)
    div_lookback  : int   Ventana para detectar divergencias OBV/CVD
    """

    def __init__(
        self,
        vol_windows : list  = None,
        spike_mult  : float = 2.5,
        dry_mult    : float = 0.4,
        div_lookback: int   = 10,
    ):
        self.windows  = vol_windows or [10, 20, 50]
        self.spike    = spike_mult
        self.dry      = dry_mult
        self.div_lb   = div_lookback

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
        out = pd.DataFrame(index=df.index)

        # ── VWAP diario ───────────────────────────────────────────────────────
        tp   = (h + l + c) / 3
        if isinstance(df.index, pd.DatetimeIndex):
            date_key = df.index.normalize()
        else:
            date_key = pd.to_datetime(df.index).normalize()

        cum_tv   = (tp * v).groupby(date_key).cumsum()
        cum_v    = v.groupby(date_key).cumsum()
        vwap     = safe_div(cum_tv, cum_v)
        out["VWAP"]                = vwap
        out["Dist_VWAP_pct"]       = safe_div(c - vwap, vwap) * 100
        out["Price_above_VWAP"]    = (c > vwap).astype(int)

        # ── OBV ───────────────────────────────────────────────────────────────
        obv_dir  = np.sign(c.diff()).replace(0, 1)
        obv      = (obv_dir * v).cumsum()
        out["OBV"]                 = obv
        out["OBV_divergence"]      = divergence(c, obv, self.div_lb)
        # OBV normalizado (z-score rolling)
        obv_mean = obv.rolling(50).mean()
        obv_std  = obv.rolling(50).std()
        out["OBV_zscore"]          = safe_div(obv - obv_mean, obv_std)

        # ── Volume Delta (proxy buy / sell con precios OHLC) ──────────────────
        rng  = (h - l).replace(0, np.nan)
        if "bid_volume" in df.columns and "ask_volume" in df.columns:
            buy_vol  = df["bid_volume"]
            sell_vol = df["ask_volume"]
        else:
            buy_vol  = v * safe_div(c - l, rng)
            sell_vol = v * safe_div(h - c, rng)

        vol_delta  = buy_vol - sell_vol
        out["Volume_delta"]        = vol_delta
        out["Volume_delta_pct"]    = safe_div(vol_delta, v) * 100   # -1 a +1

        # ── CVD (Cumulative Volume Delta) ─────────────────────────────────────
        cvd = vol_delta.cumsum()
        out["CVD"]                 = cvd
        out["CVD_divergence"]      = divergence(c, cvd, self.div_lb)

        # ── Volumen relativo multi-ventana ────────────────────────────────────
        for w in self.windows:
            out[f"Vol_ratio_{w}"]  = safe_div(v, v.rolling(w).mean())

        ref_window = 20
        vol_ma     = v.rolling(ref_window).mean()
        out["Vol_spike"]           = (v > vol_ma * self.spike).astype(int)
        out["Vol_dry_up"]          = (v < vol_ma * self.dry).astype(int)

        # ── Bid/Ask imbalance ─────────────────────────────────────────────────
        out["Bid_ask_imbalance"]   = safe_div(buy_vol - sell_vol, buy_vol + sell_vol)

        # ── Volumen en contexto de tendencia ─────────────────────────────────
        is_up   = (c > c.shift(1)).astype(float)
        is_down = (c < c.shift(1)).astype(float)
        out["Vol_up_candle"]       = v * is_up
        out["Vol_down_candle"]     = v * is_down
        # Ratio volumen en velas alcistas vs bajistas (rolling 10)
        out["Bull_bear_vol_ratio"] = safe_div(
            out["Vol_up_candle"].rolling(10).sum(),
            out["Vol_down_candle"].rolling(10).sum()
        )

        return out
