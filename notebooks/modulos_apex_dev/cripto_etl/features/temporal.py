"""
features/temporal.py
====================
Features de contexto temporal (útiles especialmente en timeframes intra-día):

  · Hora UTC, minuto, día de semana
  · Minutos desde la hora redonda (reacción en :00 y :30)
  · Sesión de mercado              (Asia / Europa / NY / Off)
  · Encoding cíclico               (sin/cos de hora y día)
  · Distancia al High/Low del día  (intra-día rolling)
  · Flag de primera / última hora  del día

Para timeframes diarios los features de hora/sesión son constantes y
el pipeline los descarta automáticamente si std==0.
"""

import numpy as np
import pandas as pd
from ._utils import safe_div


class TemporalFeatures:

    SESSIONS = {
        "Asia"  : (0,  8),
        "Europe": (8,  13),
        "NY"    : (13, 21),
        "Off"   : (21, 24),
    }

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)

        # ── Resolver índice temporal ──────────────────────────────────────────
        if isinstance(df.index, pd.DatetimeIndex):
            idx = df.index
        elif "timestamp" in df.columns:
            idx = pd.to_datetime(df["timestamp"])
        else:
            # Sin info temporal, rellenamos con constantes
            for col in ["Hour", "Minute", "Day_of_week",
                        "Mins_from_round_hour", "Hour_sin", "Hour_cos",
                        "DOW_sin", "DOW_cos"]:
                out[col] = 0
            for s in self.SESSIONS:
                out[f"Session_{s}"] = 0
            out["Dist_daily_high_pct"] = 0
            out["Dist_daily_low_pct"]  = 0
            return out

        utc = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")

        # ── Features básicas ──────────────────────────────────────────────────
        out["Hour"]                 = utc.hour
        out["Minute"]               = utc.minute
        out["Day_of_week"]          = utc.dayofweek            # 0=lunes
        out["Mins_from_round_hour"] = utc.minute + utc.second / 60
        out["Is_weekend"]           = (utc.dayofweek >= 5).astype(int)
        out["Is_month_start"]       = utc.is_month_start.astype(int)
        out["Is_month_end"]         = utc.is_month_end.astype(int)

        # ── Sesiones de mercado ────────────────────────────────────────────────
        for session, (start, end) in self.SESSIONS.items():
            out[f"Session_{session}"] = (
                (utc.hour >= start) & (utc.hour < end)
            ).astype(int)

        # ── Encoding cíclico ──────────────────────────────────────────────────
        out["Hour_sin"]             = np.sin(2 * np.pi * out["Hour"] / 24)
        out["Hour_cos"]             = np.cos(2 * np.pi * out["Hour"] / 24)
        out["DOW_sin"]              = np.sin(2 * np.pi * out["Day_of_week"] / 7)
        out["DOW_cos"]              = np.cos(2 * np.pi * out["Day_of_week"] / 7)
        out["Min_sin"]              = np.sin(2 * np.pi * out["Minute"] / 60)
        out["Min_cos"]              = np.cos(2 * np.pi * out["Minute"] / 60)

        # ── Distancia al High / Low del día (rolling cumulativo intra-día) ────
        date_key = utc.normalize()
        c = df["Close"]
        h = df["High"]
        l = df["Low"]
        daily_hi = h.groupby(date_key).cummax()
        daily_lo = l.groupby(date_key).cummin()
        out["Dist_daily_high_pct"]  = safe_div(daily_hi - c, c) * 100
        out["Dist_daily_low_pct"]   = safe_div(c - daily_lo, c) * 100
        out["Intraday_range_pct"]   = safe_div(daily_hi - daily_lo, c) * 100

        return out
