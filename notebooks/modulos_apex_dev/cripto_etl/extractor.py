"""
extractor.py
============
Descarga de datos OHLCV vía ccxt con paginación segura y over-fetch.
Sin cambios lógicos respecto al original; sólo se limpia para encajar
en la arquitectura de paquete.
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import timedelta


class ExtractorDatosCCXT:
    """
    Descarga velas OHLCV de cualquier exchange soportado por ccxt.

    Parámetros
    ----------
    exchange_id : str   Nombre del exchange (ej. 'binance', 'bybit')
    symbol      : str   Par de trading  (ej. 'BTC/USDT')
    timeframe   : str   Temporalidad    (ej. '10m', '1h', '1d')
    ventana_critica : int  Ventana (velas a cada lado) para detectar extremos
    """

    def __init__(
        self,
        exchange_id: str = "binance",
        symbol: str = "BTC/USDT",
        timeframe: str = "10m",
        ventana_critica: int = 5,
    ):
        self.exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})
        self.symbol = symbol
        self.timeframe = timeframe
        self.ventana_critica = ventana_critica

    # ──────────────────────────────────────────────────────────────────────────
    def obtener_datos(self, start_date: str, end_date: str, buffer_dias: int = 40) -> pd.DataFrame:
        """
        Descarga OHLCV con over-fetch (buffer_dias antes del start_date) para
        que los indicadores técnicos tengan suficiente historia de cálculo.

        Devuelve
        --------
        DataFrame con columnas: Open, High, Low, Close, Volume, Retorno_Log
        Índice : DatetimeIndex UTC
        """
        dt_start  = pd.to_datetime(start_date)
        dt_end    = pd.to_datetime(end_date)
        dt_fetch  = dt_start - timedelta(days=buffer_dias)

        since_ms = self.exchange.parse8601(dt_fetch.strftime("%Y-%m-%dT00:00:00Z"))
        end_ms   = self.exchange.parse8601(dt_end.strftime("%Y-%m-%dT23:59:59Z"))

        todos = []
        print(f"[ccxt] Descargando {self.symbol} | {self.timeframe} | "
              f"desde {dt_fetch.strftime('%Y-%m-%d')} hasta {dt_end.strftime('%Y-%m-%d')}")

        while since_ms < end_ms:
            velas = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, since=since_ms)
            if not velas:
                break
            validas = [v for v in velas if v[0] <= end_ms]
            todos.extend(validas)
            if not validas:
                break
            since_ms = velas[-1][0] + 1

        df = pd.DataFrame(todos, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
        df["Date"] = pd.to_datetime(df["Date"], unit="ms", utc=True)
        df.set_index("Date", inplace=True)
        df = df.apply(pd.to_numeric)
        df.drop_duplicates(inplace=True)
        df.sort_index(inplace=True)

        df["Retorno_Log"] = np.log(df["Close"] / df["Close"].shift(1))

        print(f"[ccxt] Filas descargadas: {len(df):,}  |  "
              f"Rango: {df.index[0]} → {df.index[-1]}")
        return df
