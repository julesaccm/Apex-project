"""
labeler.py
==========
Genera DOS targets binarios independientes:

  target_max : 1 si la vela es un máximo local, 0 si no
  target_min : 1 si la vela es un mínimo local, 0 si no

Métodos disponibles
--------------------
  'rolling'  (default clásico)
      Ventana centrada de 2*ventana+1 velas.
      Máximo local  → el High actual es el máximo de toda la ventana.
      Mínimo local  → el Low actual es el mínimo de toda la ventana.

  'forward'  (por retorno futuro)
      Máximo local  → el precio cae > threshold en las próximas N velas.
      Mínimo local  → el precio sube > threshold en las próximas N velas.

  'savgol'   ← NUEVO (algoritmo Savitzky-Golay del usuario)
      Suaviza High/Low con filtro SG, detecta cruces de cero en la primera
      derivada con curvatura correcta (segunda derivada), ancla al precio
      real en ±ventana//2, aplica separación mínima, alternancia estricta
      y cambio mínimo entre extremos consecutivos.
      Es el método más robusto para datos intra-día (sin lag, sin lookahead).

      Parámetros adicionales del método savgol
      ─────────────────────────────────────────
      cambio_minimo : float  Cambio % mínimo entre extremos (default 0.02 = 2 %)
      smooth_poly   : int    Orden del polinomio SG (default 3)

      Columnas extra añadidas (útiles como features)
      ──────────────────────────────────────────────
      extremo_strength : intensidad del extremo [0, 1]
      extremo_precio   : precio exacto del extremo (High o Low)
      d1               : primera derivada suavizada en el extremo
      d2               : segunda derivada suavizada en el extremo
      pct_change       : cambio % respecto al extremo anterior validado
"""

import numpy as np
import pandas as pd
from .detector_extremos import etiquetar_extremos_validados


class Labeler:
    """
    Parámetros
    ----------
    ventana_critica : int    Velas a cada lado para métodos rolling / savgol.
    forward_candles : int    Velas hacia adelante para método forward.
    threshold_pct   : float  Movimiento mínimo para forward (0.003 = 0.3 %).
    method          : str    'rolling' | 'forward' | 'savgol'
    cambio_minimo   : float  (savgol) Cambio % mínimo entre extremos (default 0.02).
    smooth_poly     : int    (savgol) Orden del polinomio SG (default 3).
    keep_meta       : bool   (savgol) Si True, conserva columnas de metadatos
                             (extremo_strength, extremo_precio, d1, d2, pct_change).
                             Estas columnas son features adicionales muy valiosas.
    """

    VALID_METHODS = ("rolling", "forward", "savgol")

    def __init__(
        self,
        ventana_critica: int   = 5,
        forward_candles: int   = 6,
        threshold_pct  : float = 0.003,
        method         : str   = "rolling",
        cambio_minimo  : float = 0.02,
        smooth_poly    : int   = 3,
        keep_meta      : bool  = True,
    ):
        if method not in self.VALID_METHODS:
            raise ValueError(f"method debe ser uno de {self.VALID_METHODS}")

        self.ventana       = ventana_critica
        self.forward       = forward_candles
        self.threshold     = threshold_pct
        self.method        = method
        self.cambio_minimo = cambio_minimo
        self.smooth_poly   = smooth_poly
        self.keep_meta     = keep_meta

    # ─────────────────────────────────────────────────────────────────────────
    # Método 1: rolling (original adaptado)
    # ─────────────────────────────────────────────────────────────────────────
    def _rolling_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        w         = self.ventana * 2 + 1
        max_local = df["High"].rolling(window=w, center=True).max()
        min_local = df["Low"].rolling(window=w, center=True).min()

        df["target_max"] = (df["High"] == max_local).astype(int)
        df["target_min"] = (df["Low"]  == min_local).astype(int)

        ambas = (df["target_max"] == 1) & (df["target_min"] == 1)
        df.loc[ambas, ["target_max", "target_min"]] = 0
        return df

    # ─────────────────────────────────────────────────────────────────────────
    # Método 2: forward-return
    # ─────────────────────────────────────────────────────────────────────────
    def _forward_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        c   = df["Close"]
        n   = self.forward
        thr = self.threshold

        fwd_h = c.rolling(n).max().shift(-n)
        fwd_l = c.rolling(n).min().shift(-n)

        up_move   = (fwd_h - c) / c
        down_move = (c - fwd_l) / c

        df["target_min"] = (up_move   >= thr).astype(int)
        df["target_max"] = (down_move >= thr).astype(int)

        ambas = (df["target_max"] == 1) & (df["target_min"] == 1)
        df.loc[ambas, ["target_max", "target_min"]] = 0
        return df

    # ─────────────────────────────────────────────────────────────────────────
    # Método 3: Savitzky-Golay (algoritmo del usuario)
    # ─────────────────────────────────────────────────────────────────────────
    def _savgol_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Llama a etiquetar_extremos_validados() y mapea sus resultados a los
        dos targets binarios estándar del pipeline.

        Mapeo:
          extremo_tipo ==  1  → target_max = 1
          extremo_tipo == -1  → target_min = 1
          extremo_tipo ==  0  → ambos = 0

        Las columnas de metadatos (extremo_strength, d1, d2, etc.) se
        conservan si keep_meta=True: son features adicionales muy útiles
        (indican la intensidad y curvatura del extremo detectado).
        """
        resultado = etiquetar_extremos_validados(
            df,
            ventana_critica = self.ventana,
            cambio_minimo   = self.cambio_minimo,
            smooth_poly     = self.smooth_poly,
        )

        df["target_max"] = (resultado["extremo_tipo"] ==  1).astype(int)
        df["target_min"] = (resultado["extremo_tipo"] == -1).astype(int)

        META_COLS = ["extremo_strength", "extremo_precio", "d1", "d2", "pct_change"]
        if self.keep_meta:
            for col in META_COLS:
                if col in resultado.columns:
                    df[col] = resultado[col]
        else:
            df.drop(columns=META_COLS, errors="ignore", inplace=True)

        return df

    # ─────────────────────────────────────────────────────────────────────────
    # Punto de entrada
    # ─────────────────────────────────────────────────────────────────────────
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Añade target_max y target_min al DataFrame.
        Con method='savgol' también añade columnas de metadatos si keep_meta=True.
        """
        df = df.copy()

        dispatch = {
            "rolling": self._rolling_labels,
            "forward": self._forward_labels,
            "savgol" : self._savgol_labels,
        }
        df = dispatch[self.method](df)

        n_max = int(df["target_max"].sum())
        n_min = int(df["target_min"].sum())
        n_tot = len(df)
        n_neu = n_tot - n_max - n_min

        meta_note = " (+metadatos SG)" if self.method == "savgol" and self.keep_meta else ""
        print(
            f"[Labeler/{self.method}{meta_note}] "
            f"target_max={n_max} ({n_max/n_tot*100:.1f}%)  "
            f"target_min={n_min} ({n_min/n_tot*100:.1f}%)  "
            f"neutral={n_neu} ({n_neu/n_tot*100:.1f}%)"
        )
        return df
