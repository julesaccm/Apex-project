# btc_ml_package

Pipeline modular para feature engineering de BTC orientado a detectar máximos y mínimos locales.

## Estructura del paquete

```
btc_ml_package/
├── __init__.py
├── extractor.py           # Descarga OHLCV via ccxt
├── labeler.py             # Dos targets binarios: target_max, target_min
├── pipeline.py            # Orquestador end-to-end
└── features/
    ├── __init__.py
    ├── _utils.py           # Utilidades compartidas
    ├── technical.py        # RSI, BB, MACD, ADX, Stoch, etc. (pandas_ta)
    ├── momentum_ext.py     # MFI, divergencias, señales extremas
    ├── trend_ext.py        # Keltner, Squeeze, Supertrend, Ichimoku, EMAs
    ├── volume_ext.py       # VWAP, OBV, CVD, ratios de volumen
    ├── candle.py           # Patrones, rachas, distancias, retornos
    ├── temporal.py         # Hora, sesión, encoding cíclico
    ├── onchain.py          # Funding rate, OI, liquidaciones (opcionales)
    ├── extreme_dist.py     # Barras/% desde el último extremo local
    └── macro_ext.py        # SP500, VIX, Gold, NDX, DXY, ETH, TNX
```

## Instalación de dependencias

```bash
pip install ccxt yfinance pandas-ta pandas numpy scikit-learn
```

## Uso rápido

```python
from btc_ml_package.pipeline import BTCPipeline

pipe = BTCPipeline(
    exchange_id="binance",
    symbol="BTC/USDT",
    timeframe="10m",          # cualquier TF soportado por ccxt
    ventana_critica=5,        # velas a cada lado para detectar extremos
    labeler_method="rolling", # "rolling" o "forward"
    include_macro=True,       # descarga SP500, VIX, Gold, etc.
)

df = pipe.run("2024-01-01", "2024-06-01")
# df tiene: OHLCV + 160+ features + target_max + target_min
```

## Familias de features (163 en total)

| Módulo | Features | Descripción |
|--------|----------|-------------|
| TechnicalFeatures | 27 | RSI, BB, MACD, ADX, Stoch, StochRSI, CCI, Williams %R, ATR, ROC, ERI, UO |
| MomentumExtFeatures | 20 | MFI, divergencias RSI/MFI/CCI, oversold/overbought, cruce StochRSI |
| TrendExtFeatures | 34 | Keltner, BB Squeeze, Supertrend, Ichimoku, EMAs 9/21/50/200 |
| VolumeExtFeatures | 19 | VWAP, OBV, CVD, Volume Delta, ratios multi-ventana |
| CandleFeatures | 36 | Hammer, Doji, Engulfing, Marubozu, rachas, distancias, retornos |
| TemporalFeatures | 20 | Hora, sesión (Asia/EU/NY), encoding cíclico, distancia al High/Low del día |
| OnChainFeatures | 17 | Funding rate, Open Interest, liquidaciones, bid-ask spread |
| ExtremeDistanceFeatures | 11 | Barras/% desde el último máximo y mínimo local |
| MacroExtFeatures | ~80 | SP500, VIX, Gold, NDX, DXY, ETH, BTC_y, TNX + correlaciones |

## Targets

| Columna | Descripción |
|---------|-------------|
| `target_max` | 1 si la vela es un máximo local, 0 si no |
| `target_min` | 1 si la vela es un mínimo local, 0 si no |

Ambos son independientes: una vela NO puede ser máximo Y mínimo simultáneamente.

## Columnas opcionales de on-chain

Si tienes datos de derivados puedes añadirlos al DataFrame antes de llamar al pipeline:

```python
df["funding_rate"]       = ...   # tasa de financiamiento
df["open_interest"]      = ...   # OI total
df["long_liquidations"]  = ...   # liquidaciones long en la vela
df["short_liquidations"] = ...   # liquidaciones short en la vela
df["bid_ask_spread"]     = ...   # spread bid-ask absoluto
```
