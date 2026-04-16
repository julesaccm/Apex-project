"""
features/technical.py
=====================
Replica y limpia agregar_indicadores_avanzados() del script original.
Usa pandas_ta para todos los indicadores ya definidos por el usuario:
RSI, Bollinger, Stoch, StochRSI, MACD, ADX, Williams %R, CCI, ATR, ROC,
Elder-Ray (ERI), Ultimate Oscillator (UO).

NO añade features nuevos aquí; eso corresponde a los módulos *_ext.
"""

import pandas as pd
import pandas_ta as ta
import warnings
warnings.filterwarnings("ignore")


class TechnicalFeatures:
    """
    Parámetros
    ----------
    Todos los periodos son configurables para facilitar búsquedas de
    hiperparámetros (Optuna, GridSearch, etc.)
    """

    def __init__(
        self,
        rsi_len     : int = 14,
        bb_len      : int = 20,
        stoch_k     : int = 9,
        stoch_d     : int = 6,
        stochrsi_len: int = 14,
        macd_fast   : int = 12,
        macd_slow   : int = 26,
        macd_sig    : int = 9,
        adx_len     : int = 14,
        willr_len   : int = 14,
        cci_len     : int = 14,
        atr_len     : int = 14,
        roc_len     : int = 10,
        eri_len     : int = 13,
        uo_fast     : int = 7,
        uo_med      : int = 14,
        uo_slow     : int = 28,
        vol_ma_len  : int = 20,
    ):
        self.params = dict(
            rsi_len=rsi_len, bb_len=bb_len,
            stoch_k=stoch_k, stoch_d=stoch_d,
            stochrsi_len=stochrsi_len,
            macd_fast=macd_fast, macd_slow=macd_slow, macd_sig=macd_sig,
            adx_len=adx_len, willr_len=willr_len,
            cci_len=cci_len, atr_len=atr_len,
            roc_len=roc_len, eri_len=eri_len,
            uo_fast=uo_fast, uo_med=uo_med, uo_slow=uo_slow,
            vol_ma_len=vol_ma_len,
        )

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula los indicadores del script original.
        Devuelve un DataFrame sólo con las nuevas columnas.
        """
        p   = self.params
        tmp = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        if not isinstance(tmp.index, pd.DatetimeIndex):
            tmp.index = pd.to_datetime(tmp.index)

        # 1. RSI
        tmp.ta.rsi(length=p["rsi_len"], append=True)

        # 2. Bollinger Bands (genera BBL, BBM, BBU, BBB, BBP)
        tmp.ta.bbands(length=p["bb_len"], append=True)

        # 3. Retorno 1 período y volumen relativo
        tmp["Retorno_1p"]        = tmp["Close"].pct_change()
        tmp["Volumen_Relativo"]  = (tmp["Volume"] /
                                    tmp["Volume"].rolling(p["vol_ma_len"]).mean())

        # 4. Stochastic
        tmp.ta.stoch(k=p["stoch_k"], d=p["stoch_d"], append=True)

        # 5. Stochastic RSI
        tmp.ta.stochrsi(length=p["stochrsi_len"], append=True)

        # 6. MACD
        tmp.ta.macd(fast=p["macd_fast"], slow=p["macd_slow"],
                    signal=p["macd_sig"], append=True)

        # 7. ADX + DMI
        tmp.ta.adx(length=p["adx_len"], append=True)

        # 8. Williams %R
        tmp.ta.willr(length=p["willr_len"], append=True)

        # 9. CCI
        tmp.ta.cci(length=p["cci_len"], append=True)

        # 10. ATR
        tmp.ta.atr(length=p["atr_len"], append=True)

        # 11. ROC
        tmp.ta.roc(length=p["roc_len"], append=True)

        # 12. Elder-Ray Index
        tmp.ta.eri(length=p["eri_len"], append=True)

        # 13. Ultimate Oscillator
        tmp.ta.uo(fast=p["uo_fast"], medium=p["uo_med"],
                  slow=p["uo_slow"], append=True)

        # Eliminamos las columnas OHLCV base para devolver sólo features
        drop = ["Open", "High", "Low", "Close", "Volume"]
        out  = tmp.drop(columns=drop, errors="ignore")
        return out
