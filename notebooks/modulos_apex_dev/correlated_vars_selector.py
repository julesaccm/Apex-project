# Librerias
from scipy.stats import kstest, norm, anderson
import pandas as pd
import numpy as np

# Función para detectar las variables más correlacionadas y seleccionar una de ellas.
def highly_correlated_vars(df=pd.DataFrame, external_df=None, value_col=None, cutoff=0.9):
    """
    ¿Qué hace la función?
    ----------
    La función highly_correlated_vars identifica pares de variables numéricas que están altamente correlacionadas (por encima de un umbral cutoff) y elimina una de cada par, según un criterio externo (por ejemplo, el IV o el AUC), o, si no se proporciona ese criterio, elimina la que tiene mayor correlación promedio con las demás.

    ¿Cómo funciona?
    ----------
    
    1. Tipo de correlación:
        - Primero, revisa si las variables tienen distribución normal usando la prueba de Kolmogorov-Smirnov.
        - Si alguna variable no es normal, usa correlación de Spearman; si todas son normales, usa Pearson.
    
    2. Cálculo de la matriz de correlación:
        - Calcula la matriz de correlación absoluta entre todas las variables.
    
    3. Bucle de eliminación:
        - Busca el par de variables con mayor correlación.
        - Si esa correlación es menor al cutoff, termina el proceso.
        - Si hay un DataFrame externo (external_df) y una columna de valor (value_col), elimina del par la variable con menor valor externo (por ejemplo, menor IV).
        - Si no hay criterio externo, elimina la variable del par que tiene mayor correlación promedio con las demás.
        - Repite el proceso hasta que no haya pares con correlación mayor al umbral.
    
    4.Devuelve:
        - Una lista de variables a eliminar para que el resto no estén altamente correlacionadas.

    ¿Para qué sirve?
    ----------

    Sirve para reducir la multicolinealidad en modelos predictivos, quedándote solo con variables relevantes y poco redundantes, usando criterios objetivos (estadísticos externos o correlación promedio).

    Parametros
    ----------
    
    df: object
        DataFrame de variables numéricas

    external_df: DataFrame con los datos externos de las variables. Se utiliza para discriminar las variables correlacionadas.
        Debe de contener solo dos columnas y sin duplicados {'Variable', value_col}.
    
    value_col: Nombre de la columna con el valor externo (por ejemplo, 'IV' o 'AUC')

    cutoff: float, entre 0 y 1 excluyentes
        Umbral de correlación
    

    """
    # Determinamos el método para el cálculo de la correlación. 
    # Si una variable no se ajusta a una distribución normal, entonces se utiliza la correlación de spearman

    not_normal_vars = []

    for i in df.columns:
        data = df[i].dropna()
        standardized = (data - data.mean())/data.std()
        pvalue = kstest(standardized,'norm').pvalue
        test = pvalue < 0.05
        if test == True:
            not_normal_vars.append(i)

    if len(not_normal_vars) > 0:
        corr_method = 'spearman'
    else:
        corr_method = 'pearson'


    corr_matrix = df.corr(method=corr_method).abs()
    cols = list(corr_matrix.columns)
    to_remove = set()
        
    while True:
        
        max_corr = 0
        max_pair = None
        
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                c = corr_matrix.loc[cols[i], cols[j]]
                if c > max_corr:
                    max_corr = c
                    max_pair = (cols[i], cols[j])
        
        if max_corr < cutoff:
            break

        if external_df is None:

            # Calculo de la correlacion promedio absoluta con las columnas restantes
            avg_corr_col1 = corr_matrix.loc[max_pair[0]].drop(max_pair[0]).mean()
            avg_corr_col2 = corr_matrix.loc[max_pair[1]].drop(max_pair[1]).mean()

            if avg_corr_col1 > avg_corr_col2:
                to_remove.add(max_pair[0])
                cols.remove(max_pair[0])
            else:
                to_remove.add(max_pair[1])
                cols.remove(max_pair[1])
            print(f'Seleccion por correlacion media: iteración {i}')
                
        else:
            if value_col is None or value_col not in external_df.columns:
                raise ValueError(
                    f"No se especifica el nombre de la columna value_col o no existe en external_df."
                )
            else:

                # Busca el valor externo de cada variable
                val1 = external_df.loc[external_df['Variable'] == max_pair[0], value_col].values[0]
                val2 = external_df.loc[external_df['Variable'] == max_pair[1], value_col].values[0]

                # Elimina la de menor valor externo
                if val1 < val2:
                    to_remove.add(max_pair[0])
                    cols.remove(max_pair[0])
                else:
                    to_remove.add(max_pair[1])
                    cols.remove(max_pair[1])
                print(f'Seleccion por criterio externo: iteración {i}')

    print('Proceso finalizado.')
    return list(to_remove)
