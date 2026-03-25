################### INICIO DEL BACKTESTER CON TRAILING STOP ###################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

class Backtester:
    """
    Clase para realizar Backtesting con Stop Trailing como estrategia de riesgo.
    """

    def __init__(self, df_test, capital_inicial=1000.0):
        # Guardamos los datos y configuraciones iniciales como atributos de la clase
        self.df_base = df_test.copy()
        self.capital_inicial = capital_inicial

    def backtest_con_trailing_stop(
            self,
            predicciones,
            tamaño_posicion=0.10, 
            stop_loss_inicial=0.05, 
            trailing_activation=0.04, 
            trailing_distancia=0.02,
            graficar=False):
        """
        Simulador avanzado con Trailing Stop.
        trailing_activation: % de ganancia necesaria para que el Trailing Stop despierte (ej. 4%).
        trailing_distancia: A qué distancia persigue el stop al precio máximo alcanzado (ej. 2%).
        """
        df_sim = self.df_base.copy()

        df_sim['Señal'] = predicciones

        # Variables locales exclusivas de esta ejecución del backtest
        capital_actual = self.capital_inicial
        posicion_abierta = False
        precio_compra = 0.0
        cantidad_btc = 0.0
        precio_max_alcanzado = 0.0
        nivel_stop_loss = 0.0
        
        historial_trades = []
        evolucion_capital = []

        for fecha, fila in df_sim.iterrows():
            precio_actual = fila['Close']
            señal = fila['Señal']

            # --- 1. REVISIÓN DE POSICIONES ABIERTAS (STOP LOSS Y TRAILING) ---
            if posicion_abierta:
                # Actualizar el precio máximo alcanzado desde que compramos
                if precio_actual > precio_max_alcanzado:
                    precio_max_alcanzado = precio_actual
                    
                    # Revisar si despertamos el Trailing Stop
                    ganancia_actual = (precio_max_alcanzado - precio_compra) / precio_compra
                    if ganancia_actual >= trailing_activation:
                        # Movemos el stop loss hacia arriba, persiguiendo al precio
                        nuevo_stop = precio_max_alcanzado * (1 - trailing_distancia)
                        if nuevo_stop > nivel_stop_loss:
                            nivel_stop_loss = nuevo_stop
                
                # Revisar si el precio cayó y tocó el Stop Loss (Inicial o Trailing)
                if precio_actual <= nivel_stop_loss:
                    precio_ejecucion = nivel_stop_loss
                    capital_actual += cantidad_btc * precio_ejecucion

                    tipo_venta = 'Venta (Trailing Stop)' if nivel_stop_loss > (precio_compra * (1 - stop_loss_inicial)) else 'Venta (Stop Inicial)'

                    historial_trades.append({'Fecha': fecha, 'Tipo': tipo_venta, 'Precio': precio_ejecucion, 'Capital': capital_actual})
                    posicion_abierta = False
                    
                    # Registramos el valor del portafolio y saltamos al siguiente día
                    evolucion_capital.append(capital_actual)
                    continue 

            # --- 2. LÓGICA DE COMPRA ---
            if señal == -1 and not posicion_abierta:
                monto_invertir = capital_actual * tamaño_posicion
                cantidad_btc = monto_invertir / precio_actual
                capital_actual -= monto_invertir
                precio_compra = precio_actual
                posicion_abierta = True
                precio_max_alcanzado = precio_actual

                # Establecemos el Stop Loss estático inicial
                nivel_stop_loss = precio_compra * (1 - stop_loss_inicial)
                
                # En la compra, el capital registrado incluye el valor del activo adquirido
                valor_portafolio_momento = capital_actual + (cantidad_btc * precio_actual)
                historial_trades.append({'Fecha': fecha, 'Tipo': 'Compra', 'Precio': precio_actual, 'Capital': valor_portafolio_momento})
                
            # --- 3. LÓGICA DE VENTA (POR SEÑAL DEL MODELO) ---
            elif señal == 1 and posicion_abierta:
                capital_actual += cantidad_btc * precio_actual
                historial_trades.append({'Fecha': fecha, 'Tipo': 'Venta (Señal Modelo)', 'Precio': precio_actual, 'Capital': capital_actual})
                posicion_abierta = False
                
            # --- 4. ACTUALIZACIÓN DEL PORTAFOLIO DIARIO ---
            valor_portafolio = capital_actual + (cantidad_btc * precio_actual if posicion_abierta else 0)
            evolucion_capital.append(valor_portafolio)
            
        # --- FIN DEL CICLO, RESULTADOS ---
        df_sim['Valor_Portafolio'] = evolucion_capital
        retorno_total = ((df_sim['Valor_Portafolio'].iloc[-1] - self.capital_inicial) / self.capital_inicial) * 100

        trades_df = pd.DataFrame(historial_trades)
        total_operaciones = len(trades_df[trades_df['Tipo'].str.contains('Venta')]) if not trades_df.empty else 0

        print(f"--- RESULTADOS DEL BACKTEST ---")
        print(f"Capital Inicial: ${self.capital_inicial:.2f} USD")
        print(f"Capital Final: ${df_sim['Valor_Portafolio'].iloc[-1]:.2f} USD")
        print(f"Retorno Total: {retorno_total:.2f}%")
        print(f"Total de operaciones cerradas: {total_operaciones}")
        
        if graficar:
            plt.figure(figsize=(10, 5))
            plt.plot(df_sim.index, df_sim['Valor_Portafolio'], color='blue', label='Curva de Capital')
            plt.title('Evolución con Trailing Stop')
            plt.ylabel('Valor en USD')
            plt.grid(True, alpha=0.3)
            plt.show()
            
        return retorno_total, trades_df
    
################### FIN DEL BACKTESTER CON TRAILING STOP ######################

### EJEMPLO DE USO ###
# simulador = Backtester(df_test, capital_inicial=1000.0)

# # Prueba 1: Conservador
# retorno_1, trades_1 = simulador.backtest_con_trailing_stop(predicciones, stop_loss_inicial=0.03, trailing_activation=0.02)

# # Prueba 2: Agresivo
# retorno_2, trades_2 = simulador.backtest_con_trailing_stop(predicciones, stop_loss_inicial=0.10, trailing_activation=0.08)