import os
import datetime
import uuid
from supabase import create_client
from src.subaccount_manager import BinanceSubaccountManager

BOT_PREFIX = "BOT_001_"


class BinanceExecutor:
    """
    Opera exclusivamente con la sub-cuenta dedicada al bot.
    Todo acceso al exchange pasa por self.sub.bot_exchange,
    que está autenticado con las credenciales de la sub-cuenta.
    """

    def __init__(self, funds: str = 'tesnet'):
        # Sub-cuenta: autenticación, balance y auditoría
        if funds == 'real':
            testnet = False
        else:
            testnet = True
            
        self.sub = BinanceSubaccountManager(testnet=testnet)

        # Alias directo al exchange de la sub-cuenta (comodidad interna)
        self.exchange = self.sub.bot_exchange

        # Supabase
        self.db = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_KEY")
        )

    # ─────────────────────────────────────────────────────────────────────────
    # IDENTIFICACIÓN DE ÓRDENES
    # ─────────────────────────────────────────────────────────────────────────
    def _generar_client_order_id(self) -> str:
        return f"{BOT_PREFIX}{uuid.uuid4().hex[:8]}"

    def _es_orden_del_bot(self, order) -> bool:
        return order.get("clientOrderId", "").startswith(BOT_PREFIX)

    # ─────────────────────────────────────────────────────────────────────────
    # AUDITORÍA DE ÓRDENES
    # ─────────────────────────────────────────────────────────────────────────
    def _auditar_ordenes_abiertas(self, symbol: str) -> tuple:
        """
        Usa el auditor del SubaccountManager para mantener un único punto
        de verificación de órdenes.
        """
        resultado = self.sub.auditar_ordenes_subcuenta(symbol, BOT_PREFIX)
        return resultado["bot"], resultado["ajenas"]

    # ─────────────────────────────────────────────────────────────────────────
    # MEMORIA DEL BOT (SUPABASE)
    # ─────────────────────────────────────────────────────────────────────────
    def _estado_inicial(self) -> dict:
        estado = {
            "id": 1,
            "posicion_abierta": False,
            "precio_compra": 0,
            "precio_max_alcanzado": 0,
            "nivel_stop_loss": 0,
            "cantidad_btc": 0,
            "trailing_activation": 0.03,
            "trailing_distancia": 0.015,
            "client_order_id": None,
        }
        self.db.table("trade_state").insert(estado).execute()
        print("Estado inicial creado en Supabase.")
        return estado

    def _cargar_estado(self) -> dict:
        try:
            res = self.db.table("trade_state").select("*").eq("id", 1).single().execute()
            return res.data if res.data else self._estado_inicial()
        except Exception as e:
            print(f"No se encontró estado en Supabase. Creando estado inicial... ({e})")
            return self._estado_inicial()

    def _guardar_estado(self, estado: dict):
        estado_limpio = {k: v for k, v in estado.items() if k != "id"}
        estado_limpio["updated_at"] = datetime.datetime.utcnow().isoformat()
        try:
            self.db.table("trade_state").update(estado_limpio).eq("id", 1).execute()
            print("Estado sincronizado con Supabase.")
        except Exception as e:
            print(f"Error al guardar estado en Supabase: {e}")

    def _registrar_operacion(self, tipo, precio=0, cantidad=0, ganancia_pct=None,
                              prob_compra=None, prob_venta=None, client_order_id=None):
        try:
            log = {"tipo": tipo, "precio": precio, "cantidad": cantidad}
            if ganancia_pct    is not None: log["ganancia_pct"]  = ganancia_pct
            if client_order_id is not None: log["client_order_id"] = client_order_id
            if prob_compra is not None or prob_venta is not None:
                log["senal_prob"] = {"prob_compra": prob_compra, "prob_venta": prob_venta}
            self.db.table("trade_log").insert(log).execute()
        except Exception as e:
            print(f"Advertencia: no se pudo registrar en trade_log: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # LECTURA DEL EXCHANGE (sub-cuenta)
    # ─────────────────────────────────────────────────────────────────────────
    def verificar_balance(self, moneda: str = "USDT") -> float:
        """Consulta el saldo libre en la sub-cuenta del bot."""
        return self.sub.obtener_balance_moneda(moneda)

    def obtener_precio_actual(self, symbol: str) -> float:
        return self.exchange.fetch_ticker(symbol)["last"]

    # ─────────────────────────────────────────────────────────────────────────
    # OPERACIONES
    # ─────────────────────────────────────────────────────────────────────────
    def ejecutar_compra_con_trailing_stop(self, symbol, tamaño_posicion,
                                          stop_loss_inicial, trailing_activation, trailing_dist,
                                          prob_compra=None, prob_venta=None):
        estado = self._cargar_estado()

        if estado["posicion_abierta"]:
            client_order_id = self._generar_client_order_id()
            print("Ya hay una posición abierta registrada en el bot. Ignorando señal de compra.")
            self._registrar_operacion("COMPRA OMITIDA",
                                      precio=self.obtener_precio_actual(symbol),
                                      cantidad=0,
                                      client_order_id=client_order_id,
                                      prob_compra=prob_compra,
                                      prob_venta=prob_venta
                                      )
            return {"exito": False}

        ordenes_bot, ordenes_ajenas = self._auditar_ordenes_abiertas(symbol)

        if ordenes_ajenas:
            # En una sub-cuenta dedicada esto es inusual; lo reportamos como alerta crítica
            print("🚨 ALERTA CRÍTICA: órdenes ajenas detectadas en la sub-cuenta del bot. "
                  "Esto no debería ocurrir si nadie más tiene las API Keys. "
                  "Compra ABORTADA hasta resolver el conflicto.")
            self._registrar_operacion("COMPRA OMITIDA",
                                      precio=self.obtener_precio_actual(symbol),
                                      cantidad=0,
                                      client_order_id=client_order_id,
                                      prob_compra=prob_compra,
                                      prob_venta=prob_venta
                                      )
            return {"exito": False, "razon": "conflicto_ordenes_subcuenta"}

        if ordenes_bot:
            print(f"El bot tiene {len(ordenes_bot)} orden(es) pendiente(s). Esperando ejecución.")
            self._registrar_operacion("COMPRA OMITIDA",
                                      precio=self.obtener_precio_actual(symbol),
                                      cantidad=0,
                                      client_order_id=client_order_id,
                                      prob_compra=prob_compra,
                                      prob_venta=prob_venta
                                      )
            return {"exito": False, "razon": "orden_bot_pendiente"}

        # Fondos disponibles en la sub-cuenta
        usdt_disponible = self.verificar_balance("USDT")
        monto_invertir  = usdt_disponible * tamaño_posicion

        if monto_invertir < 5:
            print("Saldo insuficiente en la sub-cuenta para operar (mínimo ~5 USDT).")
            self._registrar_operacion("COMPRA OMITIDA",
                                      precio=self.obtener_precio_actual(symbol),
                                      cantidad=0,
                                      client_order_id=client_order_id,
                                      prob_compra=prob_compra,
                                      prob_venta=prob_venta
                                      )
            return {"exito": False}

        print(f"Ejecutando COMPRA en sub-cuenta: {monto_invertir:.2f} USDT en {symbol}...")

        try:
            client_order_id = self._generar_client_order_id()
            orden = self.exchange.create_market_buy_order(
                symbol,
                monto_invertir / self.obtener_precio_actual(symbol),
                params={"newClientOrderId": client_order_id}
            )

            precio_ejecucion  = orden["average"] or self.obtener_precio_actual(symbol)
            cantidad_comprada = orden["filled"]

            nuevo_estado = {
                "posicion_abierta":     True,
                "precio_compra":        precio_ejecucion,
                "precio_max_alcanzado": precio_ejecucion,
                "nivel_stop_loss":      precio_ejecucion * (1 - stop_loss_inicial),
                "cantidad_btc":         cantidad_comprada,
                "trailing_activation":  trailing_activation,
                "trailing_distancia":   trailing_dist,
                "client_order_id":      client_order_id,
            }
            self._guardar_estado(nuevo_estado)
            self._registrar_operacion("COMPRA",
                                      precio=precio_ejecucion,
                                      cantidad=cantidad_comprada,
                                      client_order_id=client_order_id,
                                      prob_compra=prob_compra,
                                      prob_venta=prob_venta
                                      )
            print(f"Compra exitosa | ID: {client_order_id} | "
                  f"Precio: {precio_ejecucion} | Stop: {nuevo_estado['nivel_stop_loss']:.2f}")
            return {
                "exito": True,
                "precio_compra":    precio_ejecucion,
                "cantidad_comprada": cantidad_comprada,
                "inversion":        monto_invertir,
                "stop_loss":        nuevo_estado["nivel_stop_loss"],
                "client_order_id":  client_order_id,
            }
        except Exception as e:
            print(f"Error al ejecutar compra: {e}")
            return {"exito": False}

    def ejecutar_venta_total(self, symbol, razon="Señal del Modelo",
                             prob_compra=None, prob_venta=None):
        estado = self._cargar_estado()

        if not estado["posicion_abierta"]:
            print("No hay posiciones abiertas para vender.")
            self._registrar_operacion("SIN POSICIONES PARA VENDER",
                                      precio=self.obtener_precio_actual(symbol),
                                      cantidad=0,
                                      client_order_id=client_order_id,
                                      prob_compra=prob_compra,
                                      prob_venta=prob_venta
            )
            return {"exito": False}

        btc_disponible = self.verificar_balance(symbol.split("/")[0])

        if btc_disponible < 0.00001:
            print("⚠️  Balance BTC en sub-cuenta es ~0. Posición pudo cerrarse externamente. "
                  "Limpiando estado del bot.")
            self._guardar_estado({"posicion_abierta": False, "precio_compra": 0,
                                  "precio_max_alcanzado": 0, "nivel_stop_loss": 0,
                                  "cantidad_btc": 0, "client_order_id": None})
            self._registrar_operacion("VENTA MANUAL DETECTADA",
                                      precio=self.obtener_precio_actual(symbol),
                                      cantidad=0,
                                      client_order_id=client_order_id,
                                      prob_compra=prob_compra,
                                      prob_venta=prob_venta
            )
            return {"exito": False, "razon": "venta_detectada_como_manual"}

        print(f"Ejecutando VENTA TOTAL en sub-cuenta: {btc_disponible} {symbol} ({razon})...")

        try:
            client_order_id = self._generar_client_order_id()
            orden = self.exchange.create_market_sell_order(
                symbol, btc_disponible,
                params={"newClientOrderId": client_order_id}
            )

            precio_venta = orden["average"] or self.obtener_precio_actual(symbol)
            ganancia_pct = ((precio_venta - estado["precio_compra"]) / estado["precio_compra"]) * 100
            print(f"Venta exitosa | ID: {client_order_id} | "
                  f"Precio: {precio_venta} | Rendimiento: {ganancia_pct:.2f}%")

            self._guardar_estado({"posicion_abierta": False, "precio_compra": 0,
                                  "precio_max_alcanzado": 0, "nivel_stop_loss": 0,
                                  "cantidad_btc": 0, "client_order_id": None})
            self._registrar_operacion(
                "VENTA", precio=precio_venta, ganancia_pct=ganancia_pct,
                client_order_id=client_order_id, prob_compra=prob_compra, prob_venta=prob_venta
            )
            return {"exito": True, "precio_venta": precio_venta, "ganancia_pct": ganancia_pct}

        except Exception as e:
            print(f"Error al ejecutar venta: {e}")
            self._registrar_operacion(
                "VENTA FALLIDA", precio=precio_venta, ganancia_pct=ganancia_pct,
                client_order_id=client_order_id, prob_compra=prob_compra, prob_venta=prob_venta
            )
            return {"exito": False}

    def actualizar_trailing_stops_activos(self, symbol: str, prob_compra=None, prob_venta=None):
        estado = self._cargar_estado()
        client_order_id = self._generar_client_order_id()

        if not estado["posicion_abierta"]:
            self._registrar_operacion("ESPERA",
                            precio=self.obtener_precio_actual(symbol),
                            cantidad=0,
                            client_order_id=client_order_id,
                            prob_compra=prob_compra,
                            prob_venta=prob_venta
            )
            return

        btc_disponible = self.verificar_balance(symbol.split("/")[0])
        if btc_disponible < 0.00001:
            print("⚠️  Posición cerrada externamente en sub-cuenta. Sincronizando estado.")
            self._guardar_estado({"posicion_abierta": False, "precio_compra": 0,
                                  "precio_max_alcanzado": 0, "nivel_stop_loss": 0,
                                  "cantidad_btc": 0, "client_order_id": None})
            self._registrar_operacion("CIERRE_EXTERNO_DETECTADO",
                                      precio=self.obtener_precio_actual(symbol),
                                      cantidad=0,
                                      client_order_id=client_order_id,
                                      prob_compra=prob_compra,
                                      prob_venta=prob_venta
                                      )
            return

        precio_actual = self.obtener_precio_actual(symbol)
        print(f"Vigilando | Precio: {precio_actual} | "
              f"Stop: {estado['nivel_stop_loss']:.2f} | "
              f"Orden: {estado.get('client_order_id', '?')}")

        if precio_actual <= estado["nivel_stop_loss"]:
            print("PRECIO ALCANZÓ EL STOP LOSS.")
            self.ejecutar_venta_total(symbol, razon="Stop Loss / Trailing alcanzado")
            return

        if precio_actual > estado["precio_max_alcanzado"]:
            estado["precio_max_alcanzado"] = precio_actual
            ganancia_actual = (precio_actual - estado["precio_compra"]) / estado["precio_compra"]

            if ganancia_actual >= estado.get("trailing_activation", 0.03):
                nuevo_stop = precio_actual * (1 - estado.get("trailing_distancia", 0.015))
                if nuevo_stop > estado["nivel_stop_loss"]:
                    estado["nivel_stop_loss"] = nuevo_stop
                    print(f"Trailing Stop actualizado → {nuevo_stop:.2f}")

            self._guardar_estado(estado)