"""
features/macro_ext.py
=====================
Versión extendida del contexto macro original.

Activos descargados vía yfinance
---------------------------------
  ^GSPC   → S&P 500 Index
  ^VIX    → CBOE Volatility Index (VIX)
  GC=F    → COMEX Gold Futures
  ^NDX    → Nasdaq 100 Index
  DX-Y.NYB→ US Dollar Index (DXY)
  ETH-USD → Ethereum (proxy del mercado cripto altcoins)
  BTC-USD → Bitcoin (cross-check / cierre independiente)
  ^TNX    → US 10-Year Treasury Yield (tipos de interés)

Para cada activo se calculan
------------------------------
  _Close       precio de cierre
  _Return      retorno diario %
  _Return_5d   retorno 5 días
  _SMA_10      media simple 10 períodos
  _SMA_50      media simple 50 períodos
  _RSI_14      RSI 14 períodos
  _Volatility  desviación estándar rolling 20 períodos de retornos
  _Momentum    retorno últimos 10 períodos

Para el VIX además
------------------------------
  VIX_level         (cierre directo)
  VIX_extreme_high  flag: VIX > 30 (fear extremo)
  VIX_extreme_low   flag: VIX < 12 (complacencia extrema)
  VIX_spike         flag: VIX sube > 20% en 1 día
  VIX_term_struct   diferencia VIX/VIX3M como proxy de curva de volatilidad
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf


class MacroExtFeatures:
    """
    Parámetros
    ----------
    vol_window  : int   Ventana para volatilidad rolling (default 20)
    mom_window  : int   Ventana para momentum (default 10)
    sma_short   : int   SMA corta (default 10)
    sma_long    : int   SMA larga (default 50)
    rsi_period  : int   Período RSI (default 14)
    """

    # Activos a descargar: nombre → ticker
    ASSETS = {
        "SP500" : "^GSPC",
        "VIX"   : "^VIX",
        "Gold"  : "GC=F",
        "NDX"   : "^NDX",
        "DXY"   : "DX-Y.NYB",
        "ETH"   : "ETH-USD",
        "BTC_y" : "BTC-USD",       # segunda fuente de BTC
        "TNX"   : "^TNX",          # 10-Year yield
    }

    def __init__(
        self,
        vol_window : int = 20,
        mom_window : int = 10,
        sma_short  : int = 10,
        sma_long   : int = 50,
        rsi_period : int = 14,
    ):
        self.vol_w  = vol_window
        self.mom_w  = mom_window
        self.sma_s  = sma_short
        self.sma_l  = sma_long
        self.rsi_p  = rsi_period

    # ── RSI manual ────────────────────────────────────────────────────────────
    @staticmethod
    def _rsi(s: pd.Series, p: int) -> pd.Series:
        delta = s.diff()
        gain  = delta.clip(lower=0).ewm(com=p - 1, adjust=False).mean()
        loss  = (-delta.clip(upper=0)).ewm(com=p - 1, adjust=False).mean()
        rs    = gain / loss.replace(0, np.nan)
        return 100 - 100 / (1 + rs)

    # ── Indicadores por activo ────────────────────────────────────────────────
    def _asset_features(self, close: pd.Series, name: str) -> pd.DataFrame:
        """Genera el bloque de features para un activo dado."""
        out = pd.DataFrame(index=close.index)
        ret = close.pct_change()

        out[f"{name}_Close"]      = close
        out[f"{name}_Return"]     = ret
        out[f"{name}_Return_5d"]  = close.pct_change(5)
        out[f"{name}_SMA_{self.sma_s}"] = close.rolling(self.sma_s).mean()
        out[f"{name}_SMA_{self.sma_l}"] = close.rolling(self.sma_l).mean()
        out[f"{name}_RSI"]        = self._rsi(close, self.rsi_p)
        out[f"{name}_Volatility"] = ret.rolling(self.vol_w).std() * np.sqrt(252)
        out[f"{name}_Momentum"]   = close.pct_change(self.mom_w)
        # Precio relativo a sus SMAs
        sma_s = close.rolling(self.sma_s).mean().replace(0, np.nan)
        sma_l = close.rolling(self.sma_l).mean().replace(0, np.nan)
        out[f"{name}_above_SMA_{self.sma_s}"] = (close > sma_s).astype(int)
        out[f"{name}_above_SMA_{self.sma_l}"] = (close > sma_l).astype(int)

        return out

    # ── Features especiales del VIX ───────────────────────────────────────────
    def _vix_features(self, vix: pd.Series) -> pd.DataFrame:
        out = pd.DataFrame(index=vix.index)
        ret = vix.pct_change()
        out["VIX_level"]          = vix
        out["VIX_return"]         = ret
        out["VIX_extreme_fear"]   = (vix > 30).astype(int)
        out["VIX_extreme_low"]    = (vix < 12).astype(int)
        out["VIX_spike"]          = (ret > 0.20).astype(int)   # subida > 20% en 1 día
        out["VIX_collapse"]       = (ret < -0.15).astype(int)  # caída > 15%
        out["VIX_MA_10"]          = vix.rolling(10).mean()
        out["VIX_above_MA"]       = (vix > vix.rolling(10).mean()).astype(int)
        out["VIX_zscore"]         = (
            (vix - vix.rolling(60).mean()) /
            vix.rolling(60).std().replace(0, np.nan)
        )
        return out

    # ── Correlaciones cruzadas (SP500 vs BTC, VIX vs BTC) ────────────────────
    def _cross_features(self, macro_df: pd.DataFrame, btc_ret: pd.Series) -> pd.DataFrame:
        out = pd.DataFrame(index=macro_df.index)
        for name in ["SP500", "NDX", "VIX", "Gold", "ETH"]:
            col = f"{name}_Return"
            if col in macro_df.columns:
                ret = macro_df[col]
                # Correlación rolling 20 períodos
                out[f"Corr_BTC_{name}_20"] = btc_ret.rolling(20).corr(ret)
                # Beta simple (sensibilidad de BTC al activo)
                cov  = btc_ret.rolling(20).cov(ret)
                var  = ret.rolling(20).var().replace(0, np.nan)
                out[f"Beta_BTC_{name}_20"] = cov / var
        return out

    # ── Punto de entrada ──────────────────────────────────────────────────────
    def transform(self, df_btc: pd.DataFrame) -> pd.DataFrame:
        """
        df_btc : DataFrame OHLCV de BTC con índice DatetimeIndex.
        Devuelve DataFrame con todas las features macro.
        """
        fecha_inicio = df_btc.index.min().strftime("%Y-%m-%d")
        fecha_fin    = (df_btc.index.max() + pd.Timedelta(days=2)).strftime("%Y-%m-%d")

        macro_parts  = []
        downloaded   = {}

        for name, ticker in self.ASSETS.items():
            print(f"  [macro] Descargando {name} ({ticker})...")
            try:
                data = yf.download(ticker, start=fecha_inicio,
                                   end=fecha_fin, progress=False, auto_adjust=True)
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                close = data["Close"].squeeze()
                close.index = pd.to_datetime(close.index).tz_localize(None)
                downloaded[name] = close
            except Exception as e:
                print(f"  [macro] ERROR descargando {name}: {e}")

        # ── Alinear con el índice de BTC ──────────────────────────────────────
        btc_idx = df_btc.index
        if hasattr(btc_idx, "tz") and btc_idx.tz is not None:
            btc_dates = btc_idx.tz_convert(None).normalize()
        else:
            btc_dates = pd.to_datetime(btc_idx).normalize()

        base_df = pd.DataFrame(index=btc_idx)

        for name, close in downloaded.items():
            close_aligned = close.reindex(btc_dates.unique())
            close_aligned = close_aligned.ffill().bfill()
            # Expandir de frecuencia diaria a la frecuencia de BTC
            close_expanded = close_aligned.reindex(btc_dates).values
            s = pd.Series(close_expanded, index=btc_idx, name=name)

            if name == "VIX":
                feats = self._vix_features(s)
            else:
                feats = self._asset_features(s, name)
            macro_parts.append(feats)

        if not macro_parts:
            print("  [macro] No se descargaron datos. Devolviendo DataFrame vacío.")
            return pd.DataFrame(index=btc_idx)

        macro_df = pd.concat(macro_parts, axis=1)

        # ── Correlaciones BTC vs macro ────────────────────────────────────────
        btc_ret = df_btc["Close"].pct_change()
        cross   = self._cross_features(macro_df, btc_ret)
        macro_df = pd.concat([macro_df, cross], axis=1)

        # ── Features adicionales de riesgo / apetito ──────────────────────────
        if "SP500_Return" in macro_df.columns and "Gold_Return" in macro_df.columns:
            macro_df["Risk_on_score"] = (
                macro_df["SP500_Return"].rolling(5).mean() -
                macro_df["Gold_Return"].rolling(5).mean()
            )
        if "VIX_level" in macro_df.columns and "SP500_RSI" in macro_df.columns:
            macro_df["Fear_greed_proxy"] = (
                100 - macro_df["VIX_zscore"] * 10 +
                macro_df["SP500_RSI"]
            ) / 2

        # Forward-fill para cubrir fines de semana en TF diario
        macro_df = macro_df.ffill().bfill()

        print(f"  [macro] {macro_df.shape[1]} features macro generadas.")
        return macro_df
