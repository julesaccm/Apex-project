#################### INICIO DE LA CLASE - #####################
import ccxt
from datetime import timedelta
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta

import warnings
warnings.filterwarnings("ignore")

class ExtractorDatosCCXT:
    def __init__(self, exchange_id='binance', symbol='BTC/USDT', timeframe='1d'):#, ventana_critica=5):
        # Instanciamos el exchange dinámicamente
        self.exchange = getattr(ccxt, exchange_id)({
            'enableRateLimit': True, # Crucial para que el exchange no nos bloquee la IP
        })
        self.symbol = symbol
        # self.ventana_critica = ventana_critica
        self.timeframe = timeframe

    def obtener_datos(self, start_date, end_date, buffer_dias=40):
        # 1. Lógica de Over-fetch
        dt_start_objetivo = pd.to_datetime(start_date)
        dt_end = pd.to_datetime(end_date)
        dt_fetch_real = dt_start_objetivo - timedelta(days=buffer_dias)
        
        # Convertimos a milisegundos (formato requerido por ccxt)
        since_ms = self.exchange.parse8601(dt_fetch_real.strftime('%Y-%m-%dT00:00:00Z'))
        end_ms = self.exchange.parse8601(dt_end.strftime('%Y-%m-%dT23:59:59Z'))
        
        todos_los_datos = []
        
        # 2. Paginación segura (Bucle para extraer todo el historial necesario)
        print(f"[CCXT] Descargando {self.symbol} desde {dt_fetch_real.strftime('%Y-%m-%d')}...")
        while since_ms < end_ms:
            # fetch_ohlcv devuelve [Timestamp, Open, High, Low, Close, Volume]
            velas = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, since=since_ms)
            
            if not velas:
                break
                
            # Filtramos para no pasarnos de la fecha final
            velas_validas = [v for v in velas if v[0] <= end_ms]
            todos_los_datos.extend(velas_validas)
            
            if len(velas) > 0:
                # Avanzamos el puntero de tiempo para la siguiente llamada
                since_ms = velas[-1][0] + 1
            else:
                break
                
        # 3. Construcción y limpieza del DataFrame
        df = pd.DataFrame(todos_los_datos, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['Date'] = pd.to_datetime(df['Date'], unit='ms')
        df.set_index('Date', inplace=True)
        
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])
                    
        # Feature básica: Retornos logarítmicos
        df['Retorno_Log'] = np.log(df['Close'] / df['Close'].shift(1))
        return df

    # def etiquetar_puntos_criticos(self, df, ventana_critica=7, tolerancia_extremo=0.001):
    #     """
    #     Etiqueta máximos y mínimos locales en targets separados.
    #     Detecta tanto puntos extremos exactos como "posibles puntos extremos" cercanos.
        
    #     Parameters:
    #     -----------
    #     df : pd.DataFrame
    #         DataFrame con columnas ['High', 'Low']
    #     ventana_critica : int
    #         Número de períodos a ambos lados para buscar el extremo (default=7)
    #     tolerancia_extremo : float
    #         Tolerancia (%) para considerar un punto "próximo al extremo"
    #         Ej: 0.001 = 0.1% (detecta puntos a menos del 0.1% del extremo)
            
    #     Returns:
    #     --------
    #     pd.DataFrame con columnas:
    #         - target_max_{ventana}P: 2=máximo exacto, 1=próximo a máximo, 0=neutral
    #         - target_min_{ventana}P: 2=mínimo exacto, 1=próximo a mínimo, 0=neutral
    #     """
    #     temp_df = df.copy()
    #     ventana = ventana_critica
        
    #     # Calculamos máximos y mínimos en ventana centrada
    #     temp_df['Max_Local'] = temp_df['High'].rolling(window=ventana*2+1, center=True).max()
    #     temp_df['Min_Local'] = temp_df['Low'].rolling(window=ventana*2+1, center=True).min()
        
    #     # Inicializamos ambos targets en 0 (neutral)
    #     temp_df[f'target_max_{ventana}P'] = 0
    #     temp_df[f'target_min_{ventana}P'] = 0
        
    #     # ====== TARGET MÁXIMO ======
    #     # Máximos exactos (High == Max_Local)
    #     es_max_exacto = temp_df['High'] == temp_df['Max_Local']
    #     temp_df.loc[es_max_exacto, f'target_max_{ventana}P'] = 2  # Etiqueta fuerte
        
    #     # Puntos próximos al máximo (dentro de tolerancia)
    #     # Diferencia en %: (1 - High/Max) nos da cuán lejos está del máximo
    #     diferencia_max_pct = 1 - (temp_df['High'] / temp_df['Max_Local'])
    #     es_max_proximo = (diferencia_max_pct > 0) & (diferencia_max_pct <= tolerancia_extremo) & (~es_max_exacto)
    #     temp_df.loc[es_max_proximo, f'target_max_{ventana}P'] = 1  # Etiqueta suave
        
    #     # ====== TARGET MÍNIMO ======
    #     # Mínimos exactos (Low == Min_Local)
    #     es_min_exacto = temp_df['Low'] == temp_df['Min_Local']
    #     temp_df.loc[es_min_exacto, f'target_min_{ventana}P'] = 2  # Etiqueta fuerte
        
    #     # Puntos próximos al mínimo (dentro de tolerancia)
    #     # Diferencia en %: (Low/Min - 1) nos da cuán lejos está del mínimo
    #     diferencia_min_pct = (temp_df['Low'] / temp_df['Min_Local']) - 1
    #     es_min_proximo = (diferencia_min_pct > 0) & (diferencia_min_pct <= tolerancia_extremo) & (~es_min_exacto)
    #     temp_df.loc[es_min_proximo, f'target_min_{ventana}P'] = 1  # Etiqueta suave
        
    #     # Limpiamos columnas auxiliares
    #     # temp_df.drop(columns=['Max_Local', 'Min_Local'], inplace=True)
        
    #     return temp_df

    # def etiquetar_con_derivada_suavizada(self, df, ventana_suave=3):
    #     """
    #     Suaviza la serie primero, luego detecta máximos por primera y segunda derivada.
    #     Más robusto al ruido de los precios.
    #     """
    #     temp_df = df.copy()
        
    #     # Suavizar con media móvil exponencial (más peso en datos recientes)
    #     high_suavizado = temp_df['High'].ewm(span=ventana_suave).mean()
        
    #     # Primera derivada de la serie suavizada
    #     derivada_1 = high_suavizado.diff()
        
    #     # Segunda derivada (derivada de la derivada)
    #     derivada_2 = derivada_1.diff()
        
    #     # Máximo: cambio de signo positivo → negativo EN LA PRIMERA DERIVADA
    #     # Y segunda derivada negativa (confirma que es máximo, no mínimo)
    #     es_maximo = (
    #         (derivada_1.shift(1) > 0) &  # Estaba subiendo
    #         (derivada_1 <= 0) &           # Ahora baja o es flat
    #         (derivada_2 < 0)              # Curvatura negativa (máximo)
    #     )
        
    #     # Strength: cuánto cambio hay (mayor cambio = máximo más "fuerte")
    #     strength = abs(derivada_1.shift(1))
        

    #     # Asignar intensidad: máximos con mayor pendiente anterior = más importantes
    #     temp_df[f'target_max_dt_{ventana_suave}'] = 0
    #     temp_df.loc[es_maximo & (strength > strength.quantile(0.9)), f'target_max_dt_{ventana_suave}'] = 2
    #     temp_df.loc[es_maximo & (strength <= strength.quantile(0.9)), f'target_max_dt_{ventana_suave}'] = 1
    
    #     return temp_df

    def agregar_indicadores_avanzados(
        self,
        df,
        rsi_len=14,
        bb_len=20,
        stoch_k=9, stoch_d=6,
        stochrsi_len=14,
        macd_fast=12, macd_slow=26, macd_sig=9,
        adx_len=14,
        willr_len=14,
        cci_len=14,
        atr_len=14,
        roc_len=10,
        eri_len=13,
        uo_fast=7, uo_med=14, uo_slow=28,
        vol_ma_len=20 # Media móvil para nuestro volumen relativo
    ):
        """
        Calcula indicadores técnicos avanzados usando pandas_ta.
        Todos los periodos son parametrizables para facilitar optimizaciones futuras.
        """
        temp_df = df.copy()
        
        # Aseguramos que el índice sea datetime
        if not isinstance(temp_df.index, pd.DatetimeIndex):
            temp_df.index = pd.to_datetime(temp_df.index)

        # 1. RSI
        temp_df.ta.rsi(length=rsi_len, append=True)
        
        # 2. Bandas de Bollinger
        temp_df.ta.bbands(length=bb_len, append=True)
        
        # Retornos y Volumen Relativo manuales
        temp_df['Retorno_1d'] = temp_df['Close'].pct_change()
        temp_df['Volumen_Relativo'] = temp_df['Volume'] / temp_df['Volume'].rolling(window=vol_ma_len).mean()
        
        # 3. STOCH
        temp_df.ta.stoch(k=stoch_k, d=stoch_d, append=True)
        
        # 4. STOCHRSI
        temp_df.ta.stochrsi(length=stochrsi_len, append=True)
        
        # 5. MACD (Añadí macd_sig=9 porque pandas_ta lo usa por defecto para el histograma)
        temp_df.ta.macd(fast=macd_fast, slow=macd_slow, signal=macd_sig, append=True)
        
        # 6. ADX
        temp_df.ta.adx(length=adx_len, append=True)
        
        # 7. Williams %R
        temp_df.ta.willr(length=willr_len, append=True)
        
        # 8. CCI
        temp_df.ta.cci(length=cci_len, append=True)
        
        # 9. ATR
        temp_df.ta.atr(length=atr_len, append=True)
        
        # 10. ROC
        temp_df.ta.roc(length=roc_len, append=True)
        
        # 11. Bull/Bear Power (Elder-Ray Index)
        temp_df.ta.eri(length=eri_len, append=True)
        
        # 12. Ultimate Oscillator
        temp_df.ta.uo(fast=uo_fast, medium=uo_med, slow=uo_slow, append=True)
        
        # Limpiamos las filas iniciales que quedan con NaN por los periodos de cálculo
        temp_df.dropna(inplace=True)
        
        return temp_df

    def agregar_contexto_macro(self, df_btc):
        """
        Descarga datos del S&P 500, DXY y Oro, y los alinea con el dataset de Bitcoin,
        manejando los cierres de mercado de fin de semana.
        """
        print("Iniciando descarga de datos macroeconómicos...")
        temp_df = df_btc.copy()
        
        # Extraemos las fechas de nuestro dataset actual para descargar exactamente el mismo periodo
        fecha_inicio = temp_df.index.min()
        # Sumamos un día a la fecha final para asegurar que yfinance incluya el último día
        fecha_fin = temp_df.index.max() + pd.Timedelta(days=1) 
        
        # Diccionario con los nombres y sus tickers en Yahoo Finance
        activos_macro = {
            "SP500": "^GSPC", 
            "DXY": "DX-Y.NYB", 
            "Oro": "GC=F"
        }
        
        # Creamos un DataFrame vacío pero con las mismas fechas que Bitcoin (los 365 días del año)
        df_macro = pd.DataFrame(index=temp_df.index)
        
        for nombre, ticker in activos_macro.items():
            print(f"-> Descargando {nombre} ({ticker})...")
            data = yf.download(ticker, start=fecha_inicio, end=fecha_fin, progress=False)
            
            # Limpieza del MultiIndex de yfinance
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
                
            # Guardamos solo el precio de cierre
            df_macro[f'{nombre}_Close'] = data['Close']
        
        # --- EL TRUCO DEL FIN DE SEMANA (Forward Fill) ---
        # Rellenamos los NaN de los fines de semana con el último dato válido (Viernes)
        df_macro = df_macro.ffill()
        
        # Rellenamos hacia atrás (Back Fill) por si el día 1 del dataset fue fin de semana
        df_macro = df_macro.bfill()
        
        # Calculamos los retornos diarios de los activos macro (Esto le sirve más al modelo que el precio crudo)
        for nombre in activos_macro.keys():
            df_macro[f'{nombre}_Retorno'] = df_macro[f'{nombre}_Close'].pct_change()
            
        # Unimos la macroeconomía a nuestro dataset del Proyecto Apex
        # Como ambos comparten el mismo índice (fechas), la unión es perfecta
        temp_df = temp_df.join(df_macro)
        
        # Eliminamos la primera fila que quedará con NaN por calcular los retornos
        temp_df.dropna(inplace=True)
        
        print("\n¡Contexto macro agregado exitosamente!")
        return temp_df
    
###################### FIN DE LA CLASE BTC_DataExtractor ######################

### EJEMPLO DE USO DE LA CLASE (Pipeline Completo) ###
# # 1. Instanciar la clase (Configuramos las variables globales del proceso)
# extractor = BTC_DataExtractor(
#     fecha_inicio="2023-01-01", 
#     fecha_fin="2024-01-01", 
#     ventana_critica=5
# )

# # 2. Ejecutar las transformaciones en cadena (Pipeline)
# # Cada paso toma el DataFrame del paso anterior, lo transforma y lo devuelve

# # Paso A: Obtener el precio histórico y volumen
# df_base = extractor.obtener_datos_btc()

# # Paso B: Etiquetar máximos y mínimos (Nuestro Target)
# df_etiquetado = extractor.etiquetar_puntos_criticos(df_base)

# # Paso C: Calcular todo el análisis técnico
# df_con_indicadores = extractor.agregar_indicadores_avanzados(df_etiquetado)

# # Paso D: Enriquecer con datos macroeconómicos
# df_final = extractor.agregar_contexto_macro(df_con_indicadores)


# # 3. Revisar el resultado final
# print("\n--- Vista previa de las primeras 5 filas ---")
# print(df_final.head())

# print("\n--- Información del Dataset ---")
# print(df_final.info())

# print(f"\nDimensiones finales: {df_final.shape[0]} filas y {df_final.shape[1]} columnas")