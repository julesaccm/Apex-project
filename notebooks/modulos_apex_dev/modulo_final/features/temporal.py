"""
features/temporal.py
====================
Features de contexto temporal para velas DIARIAS.

Eliminadas vs versión intra-día
--------------------------------
  ✗ Hour, Minute, Mins_from_round_hour  → constantes en diario
  ✗ Session_Asia/Europe/NY/Off          → sin sentido en diario
  ✗ Hour_sin/cos, Min_sin/cos           → constantes en diario
  ✗ Dist_daily_high_pct/low_pct         → concepto intra-día
  ✗ Intraday_range_pct                  → concepto intra-día

Conservadas
-----------
  ✓ Day_of_week                         → lunes/viernes más volátiles en BTC
  ✓ DOW_sin/cos                         → encoding cíclico del día
  ✓ Is_weekend                          → BTC cotiza 7 días (flag relevante)
  ✓ Is_month_start / Is_month_end       → efectos de rebalanceo de portafolios
  ✓ Is_quarter_start / Is_quarter_end   → rebalanceos institucionales
  ✓ Week_of_year_sin/cos                → estacionalidad anual
  ✓ Month_sin/cos                       → estacionalidad mensual
"""

import numpy as np
import pandas as pd


class TemporalFeatures:

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)

        # ── Resolver índice temporal ──────────────────────────────────────────
        if isinstance(df.index, pd.DatetimeIndex):
            idx = df.index
        elif "timestamp" in df.columns:
            idx = pd.to_datetime(df["timestamp"])
        else:
            for col in ["Day_of_week", "DOW_sin", "DOW_cos", "Is_weekend",
                        "Is_month_start", "Is_month_end",
                        "Is_quarter_start", "Is_quarter_end",
                        "Week_sin", "Week_cos", "Month_sin", "Month_cos"]:
                out[col] = 0
            return out

        if idx.tz is not None:
            idx = idx.tz_convert("UTC")

        # ── Día de semana ─────────────────────────────────────────────────────
        out["Day_of_week"]          = idx.dayofweek           # 0=lunes, 6=domingo
        out["Is_weekend"]           = (idx.dayofweek >= 5).astype(int)
        out["DOW_sin"]              = np.sin(2 * np.pi * idx.dayofweek / 7)
        out["DOW_cos"]              = np.cos(2 * np.pi * idx.dayofweek / 7)

        # ── Inicio / fin de período ───────────────────────────────────────────
        out["Is_month_start"]       = idx.is_month_start.astype(int)
        out["Is_month_end"]         = idx.is_month_end.astype(int)
        out["Is_quarter_start"]     = idx.is_quarter_start.astype(int)
        out["Is_quarter_end"]       = idx.is_quarter_end.astype(int)

        # ── Estacionalidad anual y mensual (encoding cíclico) ─────────────────
        out["Week_sin"]             = np.sin(2 * np.pi * idx.isocalendar().week.values / 52)
        out["Week_cos"]             = np.cos(2 * np.pi * idx.isocalendar().week.values / 52)
        out["Month_sin"]            = np.sin(2 * np.pi * idx.month / 12)
        out["Month_cos"]            = np.cos(2 * np.pi * idx.month / 12)

        return out
