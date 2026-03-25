#################### INICIO DE LA CLASE BTC_DataExtractor #####################
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta

import warnings
warnings.filterwarnings("ignore")

class BTC_DataExtractor:
    """
    Clase para extraer la información de BTC, indicadores técnicos y macroeconómicos.
        - Extrae datos de Yahoo Finance.
        - Calcula indicadores técnicos avanzados usando pandas_ta.
        - Etiqueta puntos críticos (máximos y mínimos locales) para el target.
        - Agrega contexto macroeconómico (S&P 500, DXY, Oro)

    """

    def __init__(self, fecha_inicio, fecha_fin, ventana_critica=5):
        self.fecha_inicio = fecha_inicio
        self.fecha_fin = fecha_fin
        self.ventana_critica = ventana_critica

    def obtener_datos_btc(self):
        """Extrae datos de Yahoo Finance y calcula features básicas."""

        print(f"Descargando datos de BTC-USD desde {self.fecha_inicio}...")
        btc = yf.download("BTC-USD", start=self.fecha_inicio, end=self.fecha_fin)
        
        if isinstance(btc.columns, pd.MultiIndex):
            btc.columns = btc.columns.get_level_values(0)

        df = btc[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df.dropna(inplace=True)
        
        # Feature básica: Retornos logarítmicos
        df['Retorno_Log'] = np.log(df['Close'] / df['Close'].shift(1))
        return df

    def etiquetar_puntos_criticos(self, df):
        """
        Etiqueta máximos (1) y mínimos (-1) locales.
        Un punto es máximo/mínimo si es el extremo en +/- 'ventana' periodos.
        """
        # Creamos una copia para no alterar el original durante el cálculo
        temp_df = df.copy()
        
        ventana = self.ventana_critica
        # Buscamos el máximo y mínimo en la ventana centrada
        # center=True permite ver 'ventana' hacia atrás y 'ventana' hacia adelante        
        temp_df['Max_Local'] = temp_df['High'].rolling(window=ventana*2+1, center=True).max()
        temp_df['Min_Local'] = temp_df['Low'].rolling(window=ventana*2+1, center=True).min()
        
        # Inicializamos la columna Target en 0 (Neutral)
        temp_df['Target'] = 0
        
        # Si el High actual es igual al Max_Local de la ventana, es un máximo (1)
        temp_df.loc[temp_df['High'] == temp_df['Max_Local'], 'Target'] = 1
        
        # Si el Low actual es igual al Min_Local de la ventana, es un mínimo (-1)
        temp_df.loc[temp_df['Low'] == temp_df['Min_Local'], 'Target'] = -1
        
        # Limpiamos columnas auxiliares
        temp_df.drop(columns=['Max_Local', 'Min_Local'], inplace=True)
        
        return temp_df

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
            
            # Limpieza del MultiIndex de yfinance (como hicimos con BTC)
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