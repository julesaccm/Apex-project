import joblib
import pandas as pd
import numpy as np

def cargar_modelo(filepath):
    """
    Carga el diccionario del modelo exportado desde el Jupyter Notebook.
    """
    try:
        modelo_dict = joblib.load(filepath)
        print(f"Modelo cargado exitosamente desde: {filepath}")
        return modelo_dict
    except FileNotFoundError:
        raise FileNotFoundError(f"No se encontró el archivo del modelo en {filepath}. ¿Ejecutaste el script de serialización?")

def predecir_señal(modelo, X_actual, umbral):
    """
    Recibe la última vela cerrada, predice probabilidades y aplica el umbral.
    Devuelve un diccionario con:
        - señal: -1 (Compra), 1 (Venta), 0 (Mantener/Esperar)
        - prob_minimo: Probabilidad de mínimo local (compra)
        - prob_maximo: Probabilidad de máximo local (venta)
    """
    # 1. Predecir probabilidades
    probabilidades = modelo.predict_proba(X_actual)
    
    # 2. Asumimos el orden de clases que XGBoost usa internamente [0, 1, 2] -> [-1, 0, 1]
    # (Asegúrate de que este índice coincida con cómo XGBoost ordenó tus clases originalmente)
    idx_minimo = 0  # Probabilidad de clase -1
    idx_maximo = 2  # Probabilidad de clase 1
    
    prob_minimo = probabilidades[0][idx_minimo]
    prob_maximo = probabilidades[0][idx_maximo]
    
    print(f"Probabilidades detectadas -> Mínimo (Compra): {prob_minimo:.2f} | Máximo (Venta): {prob_maximo:.2f} | Umbral: {umbral}")
    
    # 3. Lógica de decisión
    if prob_minimo >= umbral:
        señal = -1
    elif prob_maximo >= umbral:
        señal = 1
    else:
        señal = 0
    
    return {
        'señal': señal,
        'prob_minimo': prob_minimo,
        'prob_maximo': prob_maximo
    }