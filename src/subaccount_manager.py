"""
src/subaccount_manager.py
─────────────────────────────────────────────────────────────────────────────
Módulo de gestión de la sub-cuenta de Binance dedicada al bot APEX.

El bot opera ÚNICAMENTE con los fondos que tenga disponibles en su sub-cuenta.
La administración de fondos (depósitos, retiros, transfers) es completamente
manual desde la app/web de Binance — este módulo no toca la cuenta principal.

PASOS MANUALES (una sola vez en Binance web/app):
  1. Perfil → Gestión de sub-cuentas → Crear sub-cuenta
  2. Habilitar trading SPOT en la sub-cuenta
  3. Generar API Keys para la sub-cuenta (permisos: Lectura + Spot Trading)
  4. Copiar esas keys en el .env:
       BINANCE_BOT_API_KEY=...
       BINANCE_BOT_SECRET=...
  5. Transferir fondos manualmente desde la app cuando quieras recargar el bot
─────────────────────────────────────────────────────────────────────────────
"""

import ccxt
import os


# ─────────────────────────────────────────────────────────────────────────────
# CLIENTE FIRMADO PARA LA API DE SUB-CUENTAS
# (ccxt no expone los endpoints de sub-cuentas, se llaman directamente)
# ─────────────────────────────────────────────────────────────────────────────

class BinanceSubaccountManager:
    """
    Autenticación, balance y auditoría de la sub-cuenta del bot.
    No requiere credenciales de la cuenta principal.
    """

    def __init__(self, testnet: bool = True):
        self.testnet = testnet

        self.bot_exchange = ccxt.binance({
            "apiKey":          os.getenv("BINANCE_BOT_API_KEY", ""),
            "secret":          os.getenv("BINANCE_BOT_SECRET", ""),
            "enableRateLimit": True,
        })
        if testnet:
            self.bot_exchange.set_sandbox_mode(True)
            print("Sub-cuenta conectada a Binance TESTNET.")
        else:
            print("Sub-cuenta conectada a Binance REAL.")

        self._validar_configuracion()

    # ─────────────────────────────────────────────────────────────────────────
    # VALIDACIÓN
    # ─────────────────────────────────────────────────────────────────────────
    def _validar_configuracion(self):
        """Falla rápido si faltan las credenciales de la sub-cuenta."""
        faltantes = [k for k in ("BINANCE_BOT_API_KEY", "BINANCE_BOT_SECRET")
                     if not os.getenv(k)]
        if faltantes:
            raise EnvironmentError(
                f"Faltan variables de entorno: {faltantes}\n"
                f"Asegúrate de haber creado la sub-cuenta en Binance y generado sus API Keys."
            )

    # ─────────────────────────────────────────────────────────────────────────
    # BALANCE
    # ─────────────────────────────────────────────────────────────────────────
    def obtener_balance_subcuenta(self) -> dict:
        """Retorna todos los activos con saldo > 0 en la sub-cuenta."""
        balance = self.bot_exchange.fetch_balance()
        activos = {
            moneda: info
            for moneda, info in balance.items()
            if isinstance(info, dict) and info.get("total", 0) > 0
        }
        print("── Balance Sub-Cuenta Bot ────────────────────────────")
        for moneda, info in activos.items():
            print(f"   {moneda}: libre={info['free']:.6f} | bloqueado={info['used']:.6f} | total={info['total']:.6f}")
        print("──────────────────────────────────────────────────────")
        return activos

    def obtener_balance_moneda(self, moneda: str = "USDT") -> float:
        """Retorna el saldo libre de una moneda específica en la sub-cuenta."""
        balance = self.bot_exchange.fetch_balance()
        libre = balance.get(moneda, {}).get("free", 0.0)
        print(f"Balance libre en sub-cuenta: {libre:.6f} {moneda}")
        return libre

    # ─────────────────────────────────────────────────────────────────────────
    # AUDITORÍA DE ÓRDENES
    # ─────────────────────────────────────────────────────────────────────────
    def auditar_ordenes_subcuenta(self, symbol: str, bot_prefix: str = "BOT_001_") -> dict:
        """
        Verifica que no haya órdenes abiertas ajenas al bot en la sub-cuenta.
        En condiciones normales esto nunca debería ocurrir, ya que la sub-cuenta
        tiene sus propias API Keys y nadie más debería operar en ella.

        Retorna: {"bot": [...], "ajenas": [...]}
        """
        try:
            abiertas = self.bot_exchange.fetch_open_orders(symbol)
            bot    = [o for o in abiertas if o.get("clientOrderId", "").startswith(bot_prefix)]
            ajenas = [o for o in abiertas if not o.get("clientOrderId", "").startswith(bot_prefix)]

            if ajenas:
                print(f"🚨 ALERTA: {len(ajenas)} orden(es) en la sub-cuenta NO creada(s) por el bot:")
                for o in ajenas:
                    print(f"   ID: {o['id']} | {o['side'].upper()} {o['amount']} @ {o.get('price', 'mkt')}")
            else:
                print("Sub-cuenta limpia: todas las órdenes abiertas son del bot.")

            return {"bot": bot, "ajenas": ajenas}
        except Exception as e:
            print(f"Error al auditar órdenes de la sub-cuenta: {e}")
            return {"bot": [], "ajenas": []}

    # ─────────────────────────────────────────────────────────────────────────
    # REPORTE DE SALUD
    # ─────────────────────────────────────────────────────────────────────────
    def reporte_completo(self, symbol: str = "BTC/USDT") -> dict:
        """Imprime un diagnóstico rápido de la sub-cuenta."""
        print("\n" + "═" * 55)
        print("  REPORTE SUB-CUENTA APEX BOT")
        print("═" * 55)
        balance = self.obtener_balance_subcuenta()
        ordenes = self.auditar_ordenes_subcuenta(symbol)
        print("═" * 55 + "\n")
        return {"balance": balance, "ordenes": ordenes}