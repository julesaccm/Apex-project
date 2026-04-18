"""
pipeline.py
===========
Orquestador end-to-end del paquete btc_ml_package.

Orden de ejecución
------------------
  1. ExtractorDatosCCXT   → OHLCV + Retorno_Log
  2. Labeler              → target_max, target_min
  3. TechnicalFeatures    → indicadores pandas_ta (RSI, BB, MACD, etc.)
  4. MomentumExtFeatures  → MFI, divergencias, señales extremas
  5. TrendExtFeatures     → Keltner, Squeeze, Supertrend, Ichimoku, EMAs
  6. VolumeExtFeatures    → VWAP, OBV, CVD, ratios de volumen
  7. CandleFeatures       → patrones, rachas, distancias, retornos
  8. TemporalFeatures     → hora, sesión, encoding cíclico
  9. OnChainFeatures      → funding rate, OI, liquidaciones (opcionales)
 10. ExtremeDistanceFeatures → barras/% desde último extremo
 11. MacroExtFeatures     → SP500, VIX, Gold, NDX, DXY, ETH, BTC_y, TNX

Salida
------
DataFrame limpio (sin NaN) con todas las features + target_max + target_min.

Uso rápido
----------
from btc_ml_package.pipeline import BTCPipeline

pipe = BTCPipeline(exchange_id="binance", symbol="BTC/USDT", timeframe="10m")
df   = pipe.run("2024-01-01", "2024-06-01")
print(df.shape)      # (n_filas, n_features + 2 targets + 5 OHLCV)
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import time

from .extractor import ExtractorDatosCCXT
from .labeler   import Labeler
from .features  import (
    TechnicalFeatures,
    MomentumExtFeatures,
    TrendExtFeatures,
    VolumeExtFeatures,
    CandleFeatures,
    TemporalFeatures,
    OnChainFeatures,
    ExtremeDistanceFeatures,
    MacroExtFeatures,
)


class BTCPipeline:
    """
    Parámetros
    ----------
    exchange_id          : str    Exchange ccxt (default 'binance')
    symbol               : str    Par de trading (default 'BTC/USDT')
    timeframe            : str    Temporalidad (default '10m')
    ventana_critica      : int    Ventana de extremos para el Labeler rolling
    labeler_method       : str    'rolling' | 'forward'
    forward_candles      : int    Velas para Labeler forward
    threshold_pct        : float  Umbral % para Labeler forward
    buffer_dias          : int    Días extra de over-fetch
    include_macro        : bool   Descargar datos macro (yfinance)
    include_onchain      : bool   Incluir módulo on-chain (requiere columnas en df)
    drop_low_variance    : bool   Eliminar columnas con varianza casi cero
    verbose              : bool   Mostrar progreso detallado
    """

    def __init__(
        self,
        exchange_id         : str   = "binance",
        symbol              : str   = "BTC/USDT",
        timeframe           : str   = "1d",
        ventana_critica     : int   = 5,
        labeler_method      : str   = "savgol",
        forward_candles     : int   = 5,
        threshold_pct       : float = 0.05,
        cambio_minimo       : float = 0.06,
        smooth_poly         : int   = 3,
        keep_meta           : bool  = True,
        buffer_dias         : int   = 60,
        include_macro       : bool  = True,
        include_onchain     : bool  = True,
        drop_low_variance   : bool  = True,
        verbose             : bool  = True,
        
    ):
        # ── Extractor ────────────────────────────────────────────────────────
        self.extractor = ExtractorDatosCCXT(
            exchange_id=exchange_id,
            symbol=symbol,
            timeframe=timeframe,
            ventana_critica=ventana_critica,
        )

        # ── Labeler ───────────────────────────────────────────────────────────
        self.labeler = Labeler(
            ventana_critica=ventana_critica,
            forward_candles=forward_candles,
            threshold_pct=threshold_pct,
            method=labeler_method,
            cambio_minimo=cambio_minimo,
            smooth_poly=smooth_poly,
            keep_meta=keep_meta,
        )

        # ── Módulos de features ───────────────────────────────────────────────
        self.technical     = TechnicalFeatures()
        self.momentum_ext  = MomentumExtFeatures()
        self.trend_ext     = TrendExtFeatures()
        self.volume_ext    = VolumeExtFeatures()
        self.candle        = CandleFeatures()
        self.temporal      = TemporalFeatures()
        self.onchain       = OnChainFeatures()
        self.extreme_dist  = ExtremeDistanceFeatures(ventana_critica=ventana_critica)
        self.macro_ext     = MacroExtFeatures()

        self.include_macro      = include_macro
        self.include_onchain    = include_onchain
        self.drop_low_var       = drop_low_variance
        self.verbose            = verbose

    # ── Utilidades internas ───────────────────────────────────────────────────
    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    @staticmethod
    def _run_module(name: str, mod, df: pd.DataFrame, verbose: bool) -> pd.DataFrame:
        """Ejecuta un módulo de features con manejo de errores."""
        t0 = time.time()
        try:
            feats = mod.transform(df)
            elapsed = time.time() - t0
            if verbose:
                print(f"    ✓ {name:<25} → {feats.shape[1]:>3} features  ({elapsed:.1f}s)")
            return feats
        except Exception as e:
            print(f"    ✗ {name:<25} → ERROR: {e}")
            return pd.DataFrame(index=df.index)

    @staticmethod
    def _drop_low_variance(df: pd.DataFrame, threshold: float = 1e-6) -> pd.DataFrame:
        """Elimina columnas con varianza < threshold (constantes o casi constantes)."""
        feature_cols = [c for c in df.columns
                        if c not in ("Open","High","Low","Close","Volume",
                                     "target_max","target_min","Retorno_Log")]
        variances    = df[feature_cols].var()
        drop_cols    = variances[variances < threshold].index.tolist()
        if drop_cols:
            print(f"    [Pipeline] Eliminando {len(drop_cols)} columnas de baja varianza: "
                  f"{drop_cols[:5]}{'...' if len(drop_cols)>5 else ''}")
            df = df.drop(columns=drop_cols)
        return df

    # ── Pipeline principal ────────────────────────────────────────────────────
    def run(
        self,
        start_date : str,
        end_date   : str,
        df_ohlcv   : pd.DataFrame = None,
    ) -> pd.DataFrame:
        """
        Ejecuta el pipeline completo.

        Parámetros
        ----------
        start_date : str          Fecha inicial 'YYYY-MM-DD'
        end_date   : str          Fecha final   'YYYY-MM-DD'
        df_ohlcv   : pd.DataFrame (opcional) Si se provee, omite la descarga ccxt

        Devuelve
        --------
        pd.DataFrame con columnas OHLCV + features + target_max + target_min
        """
        self._log("\n" + "="*65)
        self._log("  BTC ML PIPELINE — Inicio")
        self._log("="*65)

        # ── 1. Extracción ─────────────────────────────────────────────────────
        if df_ohlcv is not None:
            self._log("\n[1/11] Usando DataFrame externo provisto por el usuario.")
            df = df_ohlcv.copy()
            if "Retorno_Log" not in df.columns:
                df["Retorno_Log"] = np.log(df["Close"] / df["Close"].shift(1))
        else:
            self._log("\n[1/11] Extrayendo datos OHLCV via ccxt...")
            df = self.extractor.obtener_datos(start_date, end_date)

        # ── 2. Etiquetado ─────────────────────────────────────────────────────
        self._log("\n[2/11] Etiquetando extremos locales...")
        df = self.labeler.transform(df)

        # ── 3-10. Features técnicas ───────────────────────────────────────────
        self._log("\n[3-10/11] Calculando features técnicas...")

        # Construimos un df acumulado para que módulos posteriores
        # puedan acceder a columnas de módulos anteriores (ej: RSI_14 para divergencia)
        accumulated = df.copy()

        modules = [
            ("TechnicalFeatures",       self.technical),
            ("MomentumExtFeatures",     self.momentum_ext),
            ("TrendExtFeatures",        self.trend_ext),
            ("VolumeExtFeatures",       self.volume_ext),
            ("CandleFeatures",          self.candle),
            ("TemporalFeatures",        self.temporal),
        ]

        parts = [df]   # incluye OHLCV + targets

        for name, mod in modules:
            feats = self._run_module(name, mod, accumulated, self.verbose)
            parts.append(feats)
            # Actualizar accumulated para que el siguiente módulo vea las nuevas cols
            accumulated = pd.concat([accumulated, feats], axis=1)
            accumulated = accumulated.loc[:, ~accumulated.columns.duplicated()]

        # OnChain (opcional)
        if self.include_onchain:
            feats = self._run_module("OnChainFeatures", self.onchain,
                                     accumulated, self.verbose)
            parts.append(feats)
            accumulated = pd.concat([accumulated, feats], axis=1)

        # Extreme Distance (necesita targets ya calculados)
        feats = self._run_module("ExtremeDistanceFeatures", self.extreme_dist,
                                 accumulated, self.verbose)
        parts.append(feats)

        # ── 11. Macro ─────────────────────────────────────────────────────────
        if self.include_macro:
            self._log("\n[11/11] Descargando y calculando features macro...")
            try:
                macro_feats = self.macro_ext.transform(df)
                parts.append(macro_feats)
            except Exception as e:
                print(f"    [macro] ERROR: {e}. Se omite el módulo macro.")
        else:
            self._log("\n[11/11] Módulo macro desactivado (include_macro=False).")

        # ── Consolidar ────────────────────────────────────────────────────────
        self._log("\n  Consolidando DataFrame final...")
        result = pd.concat(parts, axis=1)
        result = result.loc[:, ~result.columns.duplicated()]

        # Columnas de metadatos SG: tienen NaN en velas no-extremo.
        # Rellenar ANTES del dropna global para no perder el 99% de filas normales.
        #   · extremo_strength, d1, d2  → 0     (sin intensidad en zona plana)
        #   · extremo_precio, pct_change → ffill (último valor extremo conocido)
        SG_META_FILL_ZERO  = ["extremo_strength", "d1", "d2"]
        SG_META_FILL_FFILL = ["extremo_precio", "pct_change"]
        for col in SG_META_FILL_ZERO:
            if col in result.columns:
                result[col] = result[col].fillna(0.0)
        for col in SG_META_FILL_FFILL:
            if col in result.columns:
                result[col] = result[col].ffill().fillna(0.0)

        n_before = len(result)
        result.dropna(inplace=True)
        n_after  = len(result)

        # ── Baja varianza ─────────────────────────────────────────────────────
        if self.drop_low_var:
            result = self._drop_low_variance(result)

        # ── Resumen final ─────────────────────────────────────────────────────
        target_cols   = [c for c in result.columns if c.startswith("target_")]
        feature_cols  = [c for c in result.columns
                         if c not in ("Open","High","Low","Close","Volume",
                                      "Retorno_Log") and c not in target_cols]

        self._log("\n" + "="*65)
        self._log(f"  ✅ PIPELINE COMPLETADO")
        self._log(f"  Filas totales descargadas   : {n_before:>7,}")
        self._log(f"  Filas válidas (sin NaN)     : {n_after:>7,}")
        self._log(f"  Features totales            : {len(feature_cols):>7,}")
        self._log(f"  Targets                     : {target_cols}")

        for tc in target_cols:
            n_pos = result[tc].sum()
            self._log(f"    {tc}: {n_pos} positivos "
                      f"({n_pos/len(result)*100:.1f}%)")
        self._log("="*65 + "\n")

        return result

    # ── Método de compatibilidad con la clase original ────────────────────────
    def obtener_datos(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Alias para compatibilidad con el script original."""
        return self.extractor.obtener_datos(start_date, end_date)

    def etiquetar_puntos_criticos(self, df: pd.DataFrame) -> pd.DataFrame:
        """Alias para compatibilidad. Devuelve df con target_max y target_min."""
        return self.labeler.transform(df)

    @staticmethod
    def feature_summary(df: pd.DataFrame) -> pd.DataFrame:
        """Estadísticas descriptivas de todas las features."""
        meta = ["Open","High","Low","Close","Volume","Retorno_Log",
                "target_max","target_min"]
        feat = df.drop(columns=meta, errors="ignore")
        return feat.describe().T.round(4)
