import ccxt
import os
import datetime
import numpy as np
from supabase import create_client

class BinanceExecutor:
    def __init__(self, api_key, api_secret, testnet=True):
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
        })

        self.db = create_client(
            os.getenv('SUPABASE_URL'),
            os.getenv('SUPABASE_KEY')
        )

        if testnet:
            self.exchange.set_sandbox_mode(True)
            print("Conectado a Binance TESTNET. Operando con fondos simulados.")
        else:
            print("ADVERTENCIA: CONECTADO A BINANCE REAL. FONDOS EN RIESGO.")

    # --- MEMORIA DEL BOT (SUPABASE) ---
    def _estado_inicial(self):
        estado = {
            'id': 1,
            'posicion_abierta': False,
            'precio_compra': 0,
            'precio_max_alcanzado': 0,
            'nivel_stop_loss': 0,
            'cantidad_btc': 0,
            'trailing_activation': 0.03,
            'trailing_distancia': 0.015,
            'updated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        self.db.table('trade_state').insert(estado).execute()
        print("Estado inicial creado en Supabase.")
        return estado

    def _cargar_estado(self):
        """Descarga el estado del bot desde Supabase."""
        try:
            print("Buscando estado en Supabase...")
            res = self.db.table('trade_state').select('*').eq('id', 1).single().execute()
            if res.data:
                print("Estado encontrado en Supabase.")
                return res.data

            print("Estado no encontrado en Supabase. Creando estado inicial...")
            return self._estado_inicial()

        except Exception as e:
            print(f"No se encontró estado en Supabase. Creando estado inicial... ({e})")
            return self._estado_inicial()

    def _guardar_estado(self, estado):
        """Guarda el estado actual del bot en Supabase."""
        estado_limpio = {k: v for k, v in estado.items() if k != 'id'}
        estado_limpio['updated_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        try:
            self.db.table('trade_state').update(estado_limpio).eq('id', 1).execute()
            print("Estado sincronizado exitosamente con Supabase.")
        except Exception as e:
            print(f"Error al guardar estado en Supabase: {e}")

    def _registrar_operacion(self, tipo, precio=0, cantidad=0, ganancia_pct=None, prob_minimo=None, prob_maximo=None, id_trade_state=None):
        """Registra cada operación/evaluación en la tabla de logs."""
        try:
            # Convertir tipos numpy a Python nativo para evitar problemas de serialización JSON
            def convertir_a_python(valor):
                if isinstance(valor, (np.floating, np.integer)):
                    return float(valor) if isinstance(valor, np.floating) else int(valor)
                return valor
            
            log = {
                'tipo': tipo,
                'precio': convertir_a_python(precio),
                'cantidad': convertir_a_python(cantidad),
            }
            if ganancia_pct is not None:
                log['ganancia_pct'] = convertir_a_python(ganancia_pct)
            if prob_minimo is not None or prob_maximo is not None:
                log['senal_prob'] = {
                    'prob_minimo': convertir_a_python(prob_minimo),
                    'prob_maximo': convertir_a_python(prob_maximo)
                }
            if id_trade_state is not None:
                log['id_trade_state'] = convertir_a_python(id_trade_state)
            self.db.table('trade_log').insert(log).execute()
        except Exception as e:
            print(f"Advertencia: no se pudo registrar la operación en trade_log: {e}")

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
            return {"exito": False}

        usdt_disponible = self.verificar_balance('USDT')
        monto_invertir = usdt_disponible * tamaño_posicion

        if monto_invertir < 10:
            print("Saldo insuficiente para operar.")
            return {"exito": False}

        print(f"Ejecutando COMPRA de {symbol} por {monto_invertir:.2f} USDT...")

        try:
            orden = self.exchange.create_market_buy_order(symbol, monto_invertir / self.obtener_precio_actual(symbol))
            precio_ejecucion = orden['average'] if orden.get('average') else self.obtener_precio_actual(symbol)
            cantidad_comprada = orden['filled']

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
            self._registrar_operacion('COMPRA', precio=precio_ejecucion, cantidad=cantidad_comprada, id_trade_state=1)
            print(f"Compra exitosa. Precio: {precio_ejecucion}. Stop inicial: {nuevo_estado['nivel_stop_loss']:.2f}")

            return {
                "exito": True,
                "precio_compra": precio_ejecucion,
                "cantidad_comprada": cantidad_comprada,
                "inversion": monto_invertir,
                "stop_loss": nuevo_estado['nivel_stop_loss']
            }

        except Exception as e:
            print(f"Error al ejecutar compra: {e}")
            return {"exito": False}

    def ejecutar_venta_total(self, symbol, razon="Señal del Modelo"):
        estado = self._cargar_estado()

        if not estado['posicion_abierta']:
            print("No hay posiciones abiertas para vender.")
            return {"exito": False}

        btc_disponible = self.verificar_balance(symbol.split('/')[0])
        print(f"Ejecutando VENTA TOTAL de {btc_disponible} {symbol} ({razon})...")

        try:
            orden = self.exchange.create_market_sell_order(symbol, btc_disponible)
            precio_venta = orden['average'] if orden.get('average') else self.obtener_precio_actual(symbol)
            ganancia_pct = ((precio_venta - estado['precio_compra']) / estado['precio_compra']) * 100
            print(f"Venta exitosa. Precio: {precio_venta}. Rendimiento del trade: {ganancia_pct:.2f}%")

            self._guardar_estado({'posicion_abierta': False, 'precio_compra': 0,
                                  'precio_max_alcanzado': 0, 'nivel_stop_loss': 0,
                                  'cantidad_btc': 0})
            self._registrar_operacion('VENTA', precio=precio_venta, ganancia_pct=ganancia_pct, id_trade_state=1)

            return {"exito": True, "precio_venta": precio_venta, "ganancia_pct": ganancia_pct}

        except Exception as e:
            print(f"Error al ejecutar venta: {e}")
            return {"exito": False}

    def actualizar_trailing_stops_activos(self, symbol):
        estado = self._cargar_estado()

        if not estado['posicion_abierta']:
            return

        precio_actual = self.obtener_precio_actual(symbol)
        print(f"Vigilando posición... Precio Actual: {precio_actual} | Stop Loss: {estado['nivel_stop_loss']:.2f}")

        if precio_actual <= estado['nivel_stop_loss']:
            print("EL PRECIO HA CAÍDO POR DEBAJO DEL STOP LOSS.")
            self.ejecutar_venta_total(symbol, razon="Stop Loss / Trailing alcanzado")
            return

        if precio_actual > estado['precio_max_alcanzado']:
            estado['precio_max_alcanzado'] = precio_actual
            ganancia_actual = (precio_actual - estado['precio_compra']) / estado['precio_compra']

            if ganancia_actual >= estado.get('trailing_activation', 0.03):
                nuevo_stop = precio_actual * (1 - estado.get('trailing_distancia', 0.015))
                if nuevo_stop > estado['nivel_stop_loss']:
                    estado['nivel_stop_loss'] = nuevo_stop
                    print(f"Trailing Stop actualizado! Nuevo nivel asegurado: {nuevo_stop:.2f}")

            self._guardar_estado(estado)

    def actualizar_registro_operacion(self, tipo, precio, cantidad, ganancia_pct=None, prob_minimo=None, prob_maximo=None, id_trade_state=None):
        self._registrar_operacion(tipo, precio, cantidad, ganancia_pct, prob_minimo, prob_maximo, id_trade_state)
