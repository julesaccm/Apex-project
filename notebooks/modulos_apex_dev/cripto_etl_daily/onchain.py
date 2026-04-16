"""
features/onchain.py
===================
Variables de mercado para velas DIARIAS.

Eliminadas vs versión intra-día
--------------------------------
  ✗ funding_rate      → tasa de perpetuos (señal intra-día, ruido en diario)
  ✗ open_interest     → relevante en intra-día; en diario pierde resolución
  ✗ long_liquidations / short_liquidations → eventos intra-día puntuales
  ✗ Funding_MA_8 / Funding_cumsum_24       → derivadas del funding eliminado
  ✗ OI_change_pct / OI_drop / OI_surge     → derivadas del OI eliminado
  ✗ Liq_*                                  → derivadas de liquidaciones

Conservadas
-----------
  ✓ bid_ask_spread / Spread_pct            → válido en diario como proxy
                                             de liquidez y riesgo de mercado
  ✓ Spread_spike                           → anomalías de liquidez diaria

Nota: si no se dispone de ninguna columna opcional, el módulo devuelve
un DataFrame vacío sin bloquear el pipeline.
"""

import numpy as np
import pandas as pd
from ._utils import safe_div


class OnChainFeatures:

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)

        # ── Bid-Ask Spread (única variable relevante en diario) ───────────────
        spread = df.get("bid_ask_spread", pd.Series(np.nan, index=df.index))
        if spread.isna().all():
            # Sin datos: devolver DataFrame vacío (no bloquear pipeline)
            return out

        sp_pct              = safe_div(spread, df["Close"]) * 100
        out["Spread_pct"]   = sp_pct
        out["Spread_spike"] = (sp_pct > sp_pct.rolling(20).mean() * 2).astype(int)
        out["Spread_MA_10"] = sp_pct.rolling(10).mean()

        return out
