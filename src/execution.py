import ccxt
import json
import os
import boto3
from botocore.exceptions import ClientError

class BinanceExecutor:
    def __init__(self, api_key, api_secret, bucket_name, testnet=True):
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
        })
        
        # Configuración de S3
        self.s3 = boto3.client('s3')
        self.bucket_name = bucket_name
        self.s3_key_name = 'trade_state.json'       # Así se llamará el archivo en S3
        self.local_temp_path = '/tmp/trade_state.json' # Única ruta permitida en Serverless
        
        if testnet:
            self.exchange.set_sandbox_mode(True)
            print("Conectado a Binance TESTNET. Operando con fondos simulados.")
        else:
            print("ADVERTENCIA: CONECTADO A BINANCE REAL. FONDOS EN RIESGO.")

    # --- NUEVA MEMORIA DEL BOT (EN LA NUBE) ---
    def _cargar_estado(self):
        """Descarga la memoria desde S3. Si no existe, la crea en la nube."""
        try:
            print(f"Buscando memoria en S3 (Bucket: {self.bucket_name})...")
            # Descargamos de S3 a nuestra carpeta temporal permitida
            self.s3.download_file(self.bucket_name, self.s3_key_name, self.local_temp_path)
            
            with open(self.local_temp_path, 'r') as f:
                return json.load(f)
                
        except ClientError as e:
            # Si el error es "404 Not Found", significa que es el Día 1 del bot
            if e.response['Error']['Code'] == "404" or e.response['Error']['Code'] == "NoSuchKey":
                print(f"Memoria no encontrada en S3. Creando estado inicial...")
                estado_base = {
                    'posicion_abierta': False, 
                    'precio_compra': 0, 
                    'precio_max_alcanzado': 0, 
                    'nivel_stop_loss': 0, 
                    'cantidad_btc': 0
                }
                self._guardar_estado(estado_base)
                return estado_base
            else:
                # Si es un error de permisos o de otro tipo, detenemos el programa
                print(f"Error crítico de AWS S3: {e}")
                raise e

    def _guardar_estado(self, estado):
        """Guarda la memoria temporalmente y la sube inmediatamente a S3."""
        # 1. Guardamos los cambios en el archivo temporal local
        with open(self.local_temp_path, 'w') as f:
            json.dump(estado, f, indent=4)
            
        # 2. Subimos el archivo a S3
        try:
            self.s3.upload_file(self.local_temp_path, self.bucket_name, self.s3_key_name)
            print(f"Memoria sincronizada exitosamente con Amazon S3.")
        except ClientError as e:
            print(f"Error al subir la memoria a S3: {e}")

    # --- LECTURA DEL EXCHANGE ---
    def verificar_balance(self, moneda='USDT'):
        """Revisa cuánto capital tienes disponible para operar."""
        balance = self.exchange.fetch_balance()
        disponible = balance[moneda]['free']
        print(f"Balance disponible: {disponible} {moneda}")

        return disponible

    def obtener_precio_actual(self, symbol):
        ticker = self.exchange.fetch_ticker(symbol)
        return ticker['last']

    # --- LÓGICA DE OPERACIÓN ---
    def ejecutar_compra_con_trailing_stop(self, symbol, tamaño_posicion, stop_loss_inicial, trailing_activation, trailing_dist):
        estado = self._cargar_estado()
        
        if estado['posicion_abierta']:
            print("Ya hay una posición abierta. Ignorando señal de compra.")
            return

        usdt_disponible = self.verificar_balance('USDT')
        monto_invertir = usdt_disponible * tamaño_posicion
        
        if monto_invertir < 10: # Límite mínimo típico de Binance
            print("Saldo insuficiente para operar.")
            return

        print(f"Ejecutando COMPRA de {symbol} por {monto_invertir:.2f} USDT...")
        
        try:
            # Enviar orden de mercado a Binance
            orden = self.exchange.create_market_buy_order(symbol, monto_invertir / self.obtener_precio_actual(symbol))
            
            # Binance devuelve los detalles de ejecución reales
            precio_ejecucion = orden['average'] if orden.get('average') else self.obtener_precio_actual(symbol)
            cantidad_comprada = orden['filled']
            
            # Guardamos la nueva memoria
            nuevo_estado = {
                'posicion_abierta': True,
                'precio_compra': precio_ejecucion,
                'precio_max_alcanzado': precio_ejecucion,
                'nivel_stop_loss': precio_ejecucion * (1 - stop_loss_inicial),
                'cantidad_btc': cantidad_comprada,
                'trailing_activation': trailing_activation,
                'trailing_distancia': trailing_dist
            }
            self._guardar_estado(nuevo_estado)
            print(f"Compra exitosa. Precio: {precio_ejecucion}. Stop inicial: {nuevo_estado['nivel_stop_loss']:.2f}")
            
        except Exception as e:
            print(f"Error al ejecutar compra: {e}")

    def ejecutar_venta_total(self, symbol, razon="Señal del Modelo"):
        estado = self._cargar_estado()
        
        if not estado['posicion_abierta']:
            print("No hay posiciones abiertas para vender.")
            return
            
        btc_disponible = self.verificar_balance(symbol.split('/')[0]) # Obtiene el balance de BTC
        
        print(f"Ejecutando VENTA TOTAL de {btc_disponible} {symbol} ({razon})...")
        
        try:
            # Enviar orden de mercado para vender todo
            orden = self.exchange.create_market_sell_order(symbol, btc_disponible)
            precio_venta = orden['average'] if orden.get('average') else self.obtener_precio_actual(symbol)
            
            # Calculamos ganancia para los logs
            ganancia_pct = ((precio_venta - estado['precio_compra']) / estado['precio_compra']) * 100
            print(f"Venta exitosa. Precio: {precio_venta}. Rendimiento del trade: {ganancia_pct:.2f}%")
            
            # Limpiamos la memoria del bot
            self._guardar_estado({'posicion_abierta': False})
            
        except Exception as e:
            print(f"Error al ejecutar venta: {e}")

    def actualizar_trailing_stops_activos(self, symbol):
        estado = self._cargar_estado()
        
        if not estado['posicion_abierta']:
            return # No hay nada que vigilar
            
        precio_actual = self.obtener_precio_actual(symbol)
        
        print(f"Vigilando posición... Precio Actual: {precio_actual} | Stop Loss: {estado['nivel_stop_loss']:.2f}")

        # 1. ¿Tocamos el Stop Loss?
        if precio_actual <= estado['nivel_stop_loss']:
            print("EL PRECIO HA CAÍDO POR DEBAJO DEL STOP LOSS.")
            self.ejecutar_venta_total(symbol, razon="Stop Loss / Trailing alcanzado")
            return

        # 2. ¿Hay un nuevo máximo? Actualizar Trailing Stop
        if precio_actual > estado['precio_max_alcanzado']:
            estado['precio_max_alcanzado'] = precio_actual
            
            ganancia_actual = (precio_actual - estado['precio_compra']) / estado['precio_compra']
            
            if ganancia_actual >= estado.get('trailing_activation', 0.03):
                nuevo_stop = precio_actual * (1 - estado.get('trailing_distancia', 0.015))
                
                if nuevo_stop > estado['nivel_stop_loss']:
                    estado['nivel_stop_loss'] = nuevo_stop
                    print(f"Trailing Stop actualizado! Nuevo nivel asegurado: {nuevo_stop:.2f}")
            
            # Guardamos los cambios
            self._guardar_estado(estado)