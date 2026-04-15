"""
features/onchain.py
===================
Variables de mercado de derivados y on-chain (OPCIONALES).
Si las columnas no están presentes en el DataFrame, se generan valores
neutros (0 o NaN) para no bloquear el pipeline.

Columnas opcionales esperadas en df
------------------------------------
  funding_rate        : tasa de financiamiento del perpetuo
  open_interest       : OI total del mercado (USD o contratos)
  long_liquidations   : volumen de liquidaciones long en la vela
  short_liquidations  : volumen de liquidaciones short en la vela
  bid_ask_spread      : spread bid-ask absoluto

Features generadas
------------------
  Funding_rate          · Funding_MA_8       · Funding_extreme_pos/neg
  OI_change_pct         · OI_drop            · OI_surge
  Liq_longs             · Liq_shorts         · Liq_ratio · Liq_total_spike
  Spread_pct            · Spread_spike
"""

import numpy as np
import pandas as pd
from ._utils import safe_div


class OnChainFeatures:

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)

        # ── Funding Rate ──────────────────────────────────────────────────────
        fr = df.get("funding_rate", pd.Series(0.0, index=df.index))
        out["Funding_rate"]         = fr
        q90 = fr.quantile(0.90) if fr.std() > 0 else 0.001
        q10 = fr.quantile(0.10) if fr.std() > 0 else -0.001
        out["Funding_extreme_pos"]  = (fr > q90).astype(int)
        out["Funding_extreme_neg"]  = (fr < q10).astype(int)
        out["Funding_MA_8"]         = fr.rolling(8).mean()
        out["Funding_cumsum_24"]    = fr.rolling(24).sum()   # presión acumulada

        # ── Open Interest ─────────────────────────────────────────────────────
        oi = df.get("open_interest", pd.Series(np.nan, index=df.index))
        if oi.isna().all():
            out["OI_change_pct"]    = 0.0
            out["OI_drop"]          = 0
            out["OI_surge"]         = 0
            out["OI_MA_ratio"]      = 1.0
        else:
            oi_chg                  = oi.pct_change()
            out["OI_change_pct"]    = oi_chg
            out["OI_drop"]          = (oi_chg < -0.03).astype(int)
            out["OI_surge"]         = (oi_chg >  0.03).astype(int)
            out["OI_MA_ratio"]      = safe_div(oi, oi.rolling(20).mean())

        # ── Liquidaciones ─────────────────────────────────────────────────────
        ll = df.get("long_liquidations",  pd.Series(0.0, index=df.index))
        sl = df.get("short_liquidations", pd.Series(0.0, index=df.index))
        total_liq = ll + sl
        out["Liq_longs"]            = ll
        out["Liq_shorts"]           = sl
        out["Liq_ratio"]            = safe_div(ll, total_liq)   # >0.5 = más longs liquidados
        out["Liq_total"]            = total_liq
        liq_ma                      = total_liq.rolling(20).mean()
        out["Liq_spike"]            = (total_liq > liq_ma * 3).astype(int)
        out["Liq_cumsum_6"]         = total_liq.rolling(6).sum()  # presión reciente

        # ── Bid-Ask Spread ────────────────────────────────────────────────────
        spread = df.get("bid_ask_spread", pd.Series(np.nan, index=df.index))
        if spread.isna().all():
            out["Spread_pct"]       = 0.0
            out["Spread_spike"]     = 0
        else:
            sp_pct                  = safe_div(spread, df["Close"]) * 100
            out["Spread_pct"]       = sp_pct
            out["Spread_spike"]     = (sp_pct > sp_pct.rolling(20).mean() * 2).astype(int)

        return out
