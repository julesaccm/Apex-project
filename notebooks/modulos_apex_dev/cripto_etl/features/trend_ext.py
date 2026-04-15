"""
features/trend_ext.py
=====================
Extensiones de tendencia/estructura NO cubiertas por technical.py:

  · Keltner Channels                 (KC_upper, KC_lower, KC_pct)
  · BB Squeeze vs Keltner            (squeeze_active)
  · BB % (posición dentro de bandas) (ya en pandas_ta como BBP, pero la normalizamos)
  · BB Width                         (volatilidad de las bandas)
  · ATR como % del precio            (volatilidad relativa)
  · MACD derivadas                   (cruce, derivada del histograma)
  · Supertrend                       (dirección, flip, distancia)
  · Ichimoku simplificado            (TK cross, posición vs nube, grosor nube)
  · EMAs multi-período               (9, 21, 50, 200) y distancias al precio
"""

import numpy as np
import pandas as pd
from ._utils import safe_div


class TrendExtFeatures:
    """
    Parámetros
    ----------
    kc_period  : int    Período Keltner  (default 20)
    kc_mult    : float  Multiplicador ATR Keltner
    bb_period  : int    Período BB (debe coincidir con TechnicalFeatures)
    bb_std     : float  Desviación estándar BB
    st_period  : int    Período ATR para Supertrend
    st_mult    : float  Multiplicador para Supertrend
    """

    def __init__(
        self,
        kc_period : int   = 20,
        kc_mult   : float = 1.5,
        bb_period : int   = 20,
        bb_std    : float = 2.0,
        st_period : int   = 10,
        st_mult   : float = 3.0,
        ema_periods: list = None,
    ):
        self.kc_period  = kc_period
        self.kc_mult    = kc_mult
        self.bb_period  = bb_period
        self.bb_std     = bb_std
        self.st_period  = st_period
        self.st_mult    = st_mult
        self.ema_periods= ema_periods or [9, 21, 50, 200]

    # ── ATR verdadero ─────────────────────────────────────────────────────────
    @staticmethod
    def _atr(h, l, c, period):
        tr = pd.concat(
            [h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1
        ).max(axis=1)
        return tr.ewm(span=period, adjust=False).mean()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        c, h, l = df["Close"], df["High"], df["Low"]
        out = pd.DataFrame(index=df.index)

        # ── ATR y ATR% ────────────────────────────────────────────────────────
        atr14 = self._atr(h, l, c, 14)
        out["ATR_pct"]             = safe_div(atr14, c) * 100

        # ── EMAs y distancias ─────────────────────────────────────────────────
        for span in self.ema_periods:
            ema = c.ewm(span=span, adjust=False).mean()
            out[f"EMA_{span}"]             = ema
            out[f"dist_EMA_{span}_pct"]    = safe_div(c - ema, c) * 100
        # Pendiente de la EMA 21 (momentum de tendencia)
        ema21 = c.ewm(span=21, adjust=False).mean()
        out["EMA_21_slope"]        = ema21.diff(3) / ema21.shift(3) * 100

        # ── Bollinger adicionales (posición y anchura normalizada) ────────────
        bb_mid = c.rolling(self.bb_period).mean()
        bb_s   = c.rolling(self.bb_period).std()
        bb_up  = bb_mid + self.bb_std * bb_s
        bb_lo  = bb_mid - self.bb_std * bb_s
        out["BB_pct_position"]     = safe_div(c - bb_lo, bb_up - bb_lo)    # 0-1
        out["BB_width_norm"]       = safe_div(bb_up - bb_lo, bb_mid) * 100 # %
        out["BB_touch_lower"]      = (c <= bb_lo * 1.002).astype(int)
        out["BB_touch_upper"]      = (c >= bb_up * 0.998).astype(int)

        # ── Keltner Channels ──────────────────────────────────────────────────
        kc_mid = c.ewm(span=self.kc_period, adjust=False).mean()
        atr_kc = self._atr(h, l, c, self.kc_period)
        kc_up  = kc_mid + self.kc_mult * atr_kc
        kc_lo  = kc_mid - self.kc_mult * atr_kc
        out["KC_upper"]            = kc_up
        out["KC_lower"]            = kc_lo
        out["KC_pct_position"]     = safe_div(c - kc_lo, kc_up - kc_lo)

        # ── Squeeze (BB dentro de KC = compresión) ────────────────────────────
        out["Squeeze_active"]      = ((bb_lo > kc_lo) & (bb_up < kc_up)).astype(int)
        out["BB_squeeze_hist"]     = bb_s - atr_kc   # <0 = squeeze activo

        # ── MACD derivadas ────────────────────────────────────────────────────
        macd_col = next((col for col in df.columns if "MACD_" in col
                         and "MACDh" not in col and "MACDs" not in col), None)
        macdh_col= next((col for col in df.columns if "MACDh_" in col), None)
        macds_col= next((col for col in df.columns if "MACDs_" in col), None)

        if macd_col and macds_col:
            macd_line = df[macd_col]
            macd_sig  = df[macds_col]
            out["MACD_cross_up"]   = ((macd_line > macd_sig) &
                                      (macd_line.shift(1) <= macd_sig.shift(1))).astype(int)
            out["MACD_cross_down"] = ((macd_line < macd_sig) &
                                      (macd_line.shift(1) >= macd_sig.shift(1))).astype(int)

        if macdh_col:
            macdh = df[macdh_col]
            out["MACD_hist_deriv"] = macdh.diff()             # aceleración
            out["MACD_hist_sign_change"] = (
                (macdh > 0) != (macdh.shift(1) > 0)
            ).astype(int)

        # ── Supertrend ────────────────────────────────────────────────────────
        atr_st  = self._atr(h, l, c, self.st_period)
        hl2     = (h + l) / 2
        ub      = hl2 + self.st_mult * atr_st
        lb      = hl2 - self.st_mult * atr_st

        st        = pd.Series(np.nan, index=df.index)
        direction = pd.Series(1, index=df.index)

        for i in range(1, len(df)):
            prev_st  = st.iloc[i - 1] if not np.isnan(st.iloc[i - 1]) else ub.iloc[i]
            prev_dir = direction.iloc[i - 1]
            ci = c.iloc[i]
            if prev_dir == 1:
                st.iloc[i]        = max(lb.iloc[i], prev_st) if ci > prev_st else ub.iloc[i]
                direction.iloc[i] = 1 if ci > st.iloc[i] else -1
            else:
                st.iloc[i]        = min(ub.iloc[i], prev_st) if ci < prev_st else lb.iloc[i]
                direction.iloc[i] = -1 if ci < st.iloc[i] else 1

        out["Supertrend_dir"]      = direction
        out["Supertrend_flip"]     = (direction != direction.shift(1)).astype(int)
        out["Supertrend_dist_pct"] = safe_div(c - st, c) * 100

        # ── Ichimoku (versión simplificada) ───────────────────────────────────
        tenkan   = (h.rolling(9).max()  + l.rolling(9).min())  / 2
        kijun    = (h.rolling(26).max() + l.rolling(26).min()) / 2
        senkou_a = ((tenkan + kijun) / 2).shift(26)
        senkou_b = ((h.rolling(52).max() + l.rolling(52).min()) / 2).shift(26)
        kumo_top = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)
        kumo_bot = pd.concat([senkou_a, senkou_b], axis=1).min(axis=1)

        out["Ichi_TK_diff"]        = safe_div(tenkan - kijun, c) * 100
        out["Ichi_TK_cross_up"]    = ((tenkan > kijun) & (tenkan.shift(1) <= kijun.shift(1))).astype(int)
        out["Ichi_TK_cross_down"]  = ((tenkan < kijun) & (tenkan.shift(1) >= kijun.shift(1))).astype(int)
        out["Ichi_above_cloud"]    = (c > kumo_top).astype(int)
        out["Ichi_below_cloud"]    = (c < kumo_bot).astype(int)
        out["Ichi_in_cloud"]       = ((c >= kumo_bot) & (c <= kumo_top)).astype(int)
        out["Ichi_cloud_thickness"]= safe_div((senkou_a - senkou_b).abs(), c) * 100
        out["Ichi_dist_cloud_pct"] = safe_div(c - kumo_top, c) * 100   # neg si debajo

        return out
