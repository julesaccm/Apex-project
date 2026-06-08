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
    Predice probabilidades y aplica el umbral de decisión.

    Retorna un dict con:
      - señal:      -1 (Compra) | 0 (Esperar) | 1 (Venta)
      - prob_compra: probabilidad de la clase -1
      - prob_venta:  probabilidad de la clase  1
      - clases:      array de clases tal como las conoce el modelo
    """
    probabilidades = modelo.predict_proba(X_actual)

    # Leer el orden REAL de clases del modelo en lugar de asumir índices fijos.
    # modelo.classes_ devuelve algo como [-1, 0, 1] o [0, 1, 2] según el entrenamiento.
    clases = list(modelo.classes_)

    if 0 not in clases or 2 not in clases:
        raise ValueError(
            f"El modelo no contiene las clases esperadas (0 y 2). "
            f"Clases encontradas: {clases}. "
            f"Revisa la codificación de etiquetas del notebook de entrenamiento."
        )

    idx_compra = clases.index(0)   # índice real de la clase "Compra"
    idx_venta  = clases.index(2)    # índice real de la clase "Venta"

    prob_compra = float(probabilidades[0][idx_compra])
    prob_venta  = float(probabilidades[0][idx_venta])

    print(
        f"Clases del modelo: {clases} | "
        f"Prob Compra (idx {idx_compra}): {prob_compra:.4f} | "
        f"Prob Venta  (idx {idx_venta}):  {prob_venta:.4f} | "
        f"Umbral: {umbral}"
    )

    if prob_compra >= umbral:
        señal = -1
    elif prob_venta >= umbral:
        señal = 1
    else:
        señal = 0

    return {
        "señal":      señal,
        "prob_compra": prob_compra,
        "prob_venta":  prob_venta,
        "clases":      clases,
    }