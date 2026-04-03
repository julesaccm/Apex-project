import os
import pandas as pd
import datetime
from dotenv import load_dotenv

# Importamos nuestros propios módulos (asumiendo que están en la carpeta src)
from src.extractor_ccxt import ExtractorDatosCCXT # Tu clase unificada
from src.model_handler import cargar_modelo, predecir_señal
from src.execution import BinanceExecutor # Clase que armaremos para enviar órdenes

# Cargamos las variables de entorno (API Keys) de forma segura
load_dotenv()

def main():
    print("="*50)
    print(f"🤖 Iniciando jApex Bot - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} CST")
    print("="*50)

    try:
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        yesterday = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")

        # 1. EXTRACCIÓN Y FEATURES
        print("[1/4] Descargando datos y calculando indicadores...")
        extractor = ExtractorDatosCCXT(exchange_id='binance', symbol='BTC/USDT', timeframe='1d')
        
        # El extractor debe encargarse del over-fetch internamente y devolver el df listo
        df_mercado = extractor.obtener_datos(
            start_date=yesterday,
            end_date=today
        ) 

        df_mercado = extractor.agregar_indicadores_avanzados(df_mercado)

        df_mercado = extractor.agregar_contexto_macro(df_mercado)

        # SEGURO DE VIDA: Nos quedamos con la última vela CERRADA (Ayer/Hoy a las 6:00 PM)
        # Ignoramos la vela actual que se está formando para evitar la "Vela Mutante"
        ultima_vela_cerrada = df_mercado.iloc[-2:-1] 

        # 2. CARGA DEL CEREBRO (MODELO)
        print("[2/4] Cargando modelo XGBoost...")
        # El archivo pkl que creaste con joblib
        modelo_dict = cargar_modelo(filepath='./models/apex_xgb_model_v1.pkl') 
        
        # 3. PREDICCIÓN
        print("[3/4] Analizando el mercado...")
        # Pasamos solo las columnas que el modelo conoce, en el orden exacto
        features_ordenadas = modelo_dict['feature_names']
        X_actual = ultima_vela_cerrada[features_ordenadas]
        
        # Obtenemos -1 (Compra), 1 (Venta) o 0 (Esperar)
        señal = predecir_señal(
            modelo=modelo_dict['modelo'], 
            X_actual=X_actual, 
            umbral=modelo_dict['umbral_decision']
        )
        
        # 4. EJECUCIÓN DEL ESCUDO Y LA ORDEN
        print(f"[4/4] Señal detectada: {señal}")
        
        # Instanciamos la conexión al exchange usando las llaves del .env
        executor = BinanceExecutor(
            api_key=os.getenv('BINANCE_TESTNET_API_KEY'),
            api_secret=os.getenv('BINANCE_TESTNET_SECRET'),
            testnet=True # Vital para no usar dinero real todavía
        )
        
        if señal == -1:
            print("Señal de COMPRA detectada. Ejecutando protocolo de entrada...")
            executor.ejecutar_compra_con_trailing_stop(
                symbol='BTC/USDT',
                tamaño_posicion=0.35,     # 35% del capital disponible
                stop_loss_inicial=0.03,   # 3%
                trailing_activation=0.03, # 3%
                trailing_dist=0.015       # 1.5%
            )
            
        elif señal == 1:
            print("Señal de VENTA detectada. Evaluando cierre de posiciones...")
            executor.ejecutar_venta_total(symbol='BTC/USDT')
            
        else:
            print("Sin señal clara. Manteniendo el capital de reserva intacto. Esperando al próximo cierre diario.")
            # Ejecutor revisa si hay posiciones abiertas que requieran actualizar su Trailing Stop
            executor.actualizar_trailing_stops_activos(symbol='BTC/USDT')

    except Exception as e:
        print(f"ERROR CRÍTICO EN LA EJECUCIÓN: {e}")
        # Aquí en el futuro podríamos agregar un envío de mensaje a Telegram/WhatsApp avisando del fallo

if __name__ == "__main__":
    main()