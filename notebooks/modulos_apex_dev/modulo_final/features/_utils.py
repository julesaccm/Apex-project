"""
features/_utils.py
==================
Funciones utilitarias compartidas entre todos los módulos de features.
"""
import numpy as np
import pandas as pd


def safe_div(a: pd.Series, b: pd.Series, fill: float = 0.0) -> pd.Series:
    """División segura evitando divisiones por cero / NaN."""
    return a.div(b.replace(0, np.nan)).fillna(fill)


def rsi_manual(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI Wilder (EWM) sin dependencias externas."""
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    return 100 - 100 / (1 + safe_div(gain, loss))


def divergence(price: pd.Series, oscillator: pd.Series, lookback: int = 20) -> pd.Series:
    """
    Detecta divergencias entre precio y oscilador.
    +1 = alcista (precio nuevo mínimo, oscilador no)
    -1 = bajista (precio nuevo máximo, oscilador no)
     0 = sin divergencia
    """
    pmin = price.rolling(lookback).min()
    pmax = price.rolling(lookback).max()
    omin = oscillator.rolling(lookback).min()
    omax = oscillator.rolling(lookback).max()
    bull = ((price == pmin) & (oscillator > omin)).astype(int)
    bear = ((price == pmax) & (oscillator < omax)).astype(int)
    return bull - bear


def streak(binary_series: pd.Series) -> pd.Series:
    """Cuenta racha consecutiva de True/1 (se reinicia a 0 en False/0)."""
    s = binary_series.astype(float).copy()
    for i in range(1, len(s)):
        s.iloc[i] = s.iloc[i - 1] + 1 if binary_series.iloc[i] else 0
    return s