"""
features/momentum_ext.py
========================
Extensiones de momentum NO cubiertas por technical.py:

  · MFI (Money Flow Index)           con divergencia y señales extremas
  · RSI divergencia                  (precio vs RSI)
  · MFI divergencia                  (precio vs MFI)
  · Flags oversold / overbought      para RSI, StochRSI, CCI, Williams %R
  · Señales de cruce StochRSI        (K cruza D)
  · Composite Momentum Score         (suma ponderada de señales extremas)

Nota: RSI, CCI y Williams %R ya son calculados por TechnicalFeatures.
      Aquí sólo añadimos las capas de señal/divergencia encima.
"""

import numpy as np
import pandas as pd
import pandas_ta as ta_lib
from ._utils import safe_div, divergence


class MomentumExtFeatures:
    """
    Parámetros
    ----------
    mfi_len         : int   Período del MFI  (default 14)
    divergence_lb   : int   Ventana para detectar divergencias
    rsi_os / rsi_ob : float Umbrales oversold / overbought del RSI
    """

    def __init__(
        self,
        mfi_len      : int   = 14,
        divergence_lb: int   = 20,
        rsi_os       : float = 30.0,
        rsi_ob       : float = 70.0,
        rsi_extreme_os: float = 20.0,
        rsi_extreme_ob: float = 80.0,
    ):
        self.mfi_len   = mfi_len
        self.div_lb    = divergence_lb
        self.rsi_os    = rsi_os
        self.rsi_ob    = rsi_ob
        self.rsi_xos   = rsi_extreme_os
        self.rsi_xob   = rsi_extreme_ob

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        df debe contener: Open, High, Low, Close, Volume
        y opcionalmente columnas de TechnicalFeatures (RSI_14, CCI_14, etc.)
        """
        c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
        out = pd.DataFrame(index=df.index)

        # ── MFI ──────────────────────────────────────────────────────────────
        tp      = (h + l + c) / 3
        tp_diff = tp.diff()
        raw_mf  = tp * v
        pos_mf  = raw_mf.where(tp_diff > 0, 0.0).rolling(self.mfi_len).sum()
        neg_mf  = raw_mf.where(tp_diff < 0, 0.0).rolling(self.mfi_len).sum()
        mfi     = 100 - 100 / (1 + safe_div(pos_mf, neg_mf))
        out["MFI"]              = mfi
        out["MFI_oversold"]     = (mfi < 20).astype(int)
        out["MFI_overbought"]   = (mfi > 80).astype(int)
        out["MFI_divergence"]   = divergence(c, mfi, self.div_lb)

        # ── RSI divergencia y señales (usa RSI ya calculado si existe) ────────
        rsi_col = next((col for col in df.columns if col.startswith("RSI_")), None)
        if rsi_col:
            rsi = df[rsi_col]
        else:
            # Cálculo de respaldo
            delta = c.diff()
            gain  = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
            loss  = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
            rsi   = 100 - 100 / (1 + safe_div(gain, loss))

        out["RSI_divergence"]       = divergence(c, rsi, self.div_lb)
        out["RSI_oversold"]         = (rsi < self.rsi_os).astype(int)
        out["RSI_overbought"]       = (rsi > self.rsi_ob).astype(int)
        out["RSI_extreme_oversold"] = (rsi < self.rsi_xos).astype(int)
        out["RSI_extreme_overbought"]=(rsi > self.rsi_xob).astype(int)

        # ── CCI señales (usa CCI ya calculado) ────────────────────────────────
        cci_col = next((col for col in df.columns if col.startswith("CCI_")), None)
        if cci_col:
            cci = df[cci_col]
            out["CCI_extreme_low"]  = (cci < -150).astype(int)
            out["CCI_extreme_high"] = (cci > 150).astype(int)
            out["CCI_divergence"]   = divergence(c, cci, self.div_lb)

        # ── Williams %R señales ───────────────────────────────────────────────
        willr_col = next((col for col in df.columns if col.startswith("WILLR_")), None)
        if willr_col:
            wr = df[willr_col]
            out["WR_oversold"]  = (wr < -90).astype(int)
            out["WR_overbought"]= (wr > -10).astype(int)

        # ── StochRSI cruce K/D ────────────────────────────────────────────────
        k_col = next((c_ for c_ in df.columns if "STOCHRSIk" in c_), None)
        d_col = next((c_ for c_ in df.columns if "STOCHRSId" in c_), None)
        if k_col and d_col:
            k, d = df[k_col], df[d_col]
            out["StochRSI_cross_up"]  = ((k > d) & (k.shift(1) <= d.shift(1))).astype(int)
            out["StochRSI_cross_down"]= ((k < d) & (k.shift(1) >= d.shift(1))).astype(int)
            out["StochRSI_oversold"]  = (k < 20).astype(int)
            out["StochRSI_overbought"]= (k > 80).astype(int)

        # ── Composite Momentum Score (suma de señales alcistas / bajistas) ─────
        bull_signals = sum(
            out[col] for col in out.columns
            if any(x in col for x in ["oversold", "cross_up", "extreme_low"])
            and col in out.columns
        )
        bear_signals = sum(
            out[col] for col in out.columns
            if any(x in col for x in ["overbought", "cross_down", "extreme_high"])
            and col in out.columns
        )
        out["Momentum_bull_score"] = bull_signals
        out["Momentum_bear_score"] = bear_signals

        return out
