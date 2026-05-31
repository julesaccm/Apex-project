import os
import pandas as pd
import datetime
from dotenv import load_dotenv
import requests

# Importamos nuestros propios módulos (asumiendo que están en la carpeta src)
from src.extractor_ccxt import ExtractorDatosCCXT
from src.model_handler import cargar_modelo, predecir_señal
from src.execution import BinanceExecutor

# Cargamos las variables de entorno (API Keys) de forma segura
load_dotenv()

def enviar_telegram(mensaje):
    """Envía una alerta directa a tu celular vía Telegram."""
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')

    if not token or not chat_id:
        print("Credenciales de Telegram no encontradas. Mensaje omitido.")
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": mensaje,
        "parse_mode": "Markdown" # Permite usar negritas y emojis
    }

    try:
        requests.post(url, json=payload, timeout=5)
    except Exception as e:
        print(f"Error de red al intentar enviar Telegram: {e}")


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
        df_mercado = extractor.obtener_datos(start_date=yesterday, end_date=today)
        df_mercado = extractor.agregar_indicadores_avanzados(df_mercado)
        df_mercado = extractor.agregar_contexto_macro(df_mercado)

        # SEGURO DE VIDA: Última vela CERRADA (evitar "Vela Mutante")
        ultima_vela_cerrada = df_mercado.iloc[-2:-1]

        # 2. CARGA DEL MODELO
        print("[2/4] Cargando modelo XGBoost...")
        modelo_dict = cargar_modelo(filepath='./models/apex_xgb_model_v1.pkl')

        # 3. PREDICCIÓN
        print("[3/4] Analizando el mercado...")
        # Pasamos solo las columnas que el modelo conoce, en el orden exacto
        features_ordenadas = modelo_dict['feature_names']
        X_actual = ultima_vela_cerrada[features_ordenadas]

        # Obtenemos un diccionario con: señal, prob_minimo, prob_maximo
        prediccion = predecir_señal(
            modelo=modelo_dict['modelo'],
            X_actual=X_actual,
            umbral=modelo_dict['umbral_decision']
        )
        
        señal = prediccion['señal']
        prob_minimo = prediccion['prob_minimo']
        prob_maximo = prediccion['prob_maximo']

        # 4. EJECUCIÓN
        print(f"[4/4] Señal detectada: {señal}")

        # Sin bucket_name — el estado ahora vive en Supabase
        executor = BinanceExecutor(
            api_key=os.getenv('BINANCE_TESTNET_API_KEY'),
            api_secret=os.getenv('BINANCE_TESTNET_SECRET'),
            testnet=True
        )

        if señal == -1:
            print("Señal de COMPRA detectada. Ejecutando protocolo...")
            resultado = executor.ejecutar_compra_con_trailing_stop(
                symbol='BTC/USDT',
                tamaño_posicion=0.35,
                stop_loss_inicial=0.03,
                trailing_activation=0.03,
                trailing_dist=0.015
            )

            if resultado.get("exito"):
                # Extraemos los datos del diccionario
                precio    = resultado["precio_compra"]
                cantidad  = resultado["cantidad_comprada"]
                inversion = resultado["inversion"]
                stop      = resultado["stop_loss"]
                mensaje = (
                    f"🟢 *COMPRA EJECUTADA*\n"
                    f"💰 Precio Entrada: *${precio:.2f}*\n"
                    f"🪙 Cantidad BTC: *{cantidad:.4f}*\n"
                    f"💰 Inversión: *${inversion:.2f}*\n"
                    f"🛡️ Stop Loss Inicial: *${stop:.2f}*"
                )
                enviar_telegram(mensaje)
            else:
                enviar_telegram("⚠️ *SEÑAL DE COMPRA OMITIDA*\n(Posición abierta, saldo insuficiente o error de red).")

        elif señal == 1:
            print("Señal de VENTA detectada. Evaluando cierre...")
            resultado = executor.ejecutar_venta_total(symbol='BTC/USDT')

            if resultado.get("exito"):
                precio   = resultado["precio_venta"]
                ganancia = resultado["ganancia_pct"]
                emoji    = "📈" if ganancia > 0 else "📉"
                mensaje = (
                    f"🔴 *VENTA EJECUTADA*\n"
                    f"💰 Precio Salida: *${precio:.2f}*\n"
                    f"{emoji} Rendimiento: *{ganancia:.2f}%*"
                )
                enviar_telegram(mensaje)

        else:
            print("Sin señal clara. Manteniendo el capital de reserva intacto.")
            # Registra la decisión de espera en trade_log con sus probabilidades
            precio_actual = ultima_vela_cerrada['Close'].values[0]
            executor._registrar_operacion(
                tipo='ESPERA', 
                precio=precio_actual,
                prob_minimo=prob_minimo,
                prob_maximo=prob_maximo
            )
            
            # Ejecutor revisa si hay posiciones abiertas que requieran actualizar su Trailing Stop
            executor.actualizar_trailing_stops_activos(symbol='BTC/USDT')
            enviar_telegram("⚪ *REPORTE DIARIO: SIN CAMBIOS*\nEl bot evaluó el mercado y decidió mantenerse a la espera. Capital seguro.")

    except Exception as e:
        print(f"ERROR CRÍTICO EN LA EJECUCIÓN: {e}")
        # Notificación de emergencia
        enviar_telegram(f"🚨 *ERROR CRÍTICO EN jAPEX BOT*\nRevisa los logs. Detalle: `{e}`")


if __name__ == "__main__":
    main()


# --- ENCHUFE PARA AWS LAMBDA ---
def handler(event, context):
    """Puerta de entrada para AWS Lambda."""
    main()
    return {"statusCode": 200, "body": "Ejecución de Apex Bot finalizada."}
