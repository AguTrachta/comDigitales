# src/ofdm_tf/mapping.py

"""
Módulo para el mapeo de símbolos a subportadoras, siguiendo
la formulación de G. Lindell.
"""
import numpy as np
from . import params as p

def get_baseband_freq_indices():
    """
    Calcula los índices de frecuencia de paso bajo (g_k) para cada subportadora de datos.

    La función sigue las Ecuaciones (1.9) y (1.10) de Lindell.
    
    Returns:
        np.ndarray: Un array de tamaño K con los índices g_k.
        int: El valor de g_0 (el índice g_k más bajo).
        int: El valor de g_{K-1} (el índice g_k más alto).
    """
    k = np.arange(p.K)  # Índices de 0 a K-1
    
    if p.K % 2 == 0:  # Caso K par
        k_rc = (p.K - 2) // 2
        g_k = k - k_rc
    else:  # Caso K impar
        k_rc = (p.K - 1) // 2
        g_k = k - k_rc
        
    g0 = g_k[0]
    gK_minus_1 = g_k[-1]
    
    # Verificación de consistencia
    assert g_k[k_rc] == 0, "El índice de la portadora de referencia no es 0."
    
    return g_k, g0, gK_minus_1

def map_symbols_to_ifft_input(ak_symbols, dtype=complex): 
    """
    Construye el vector de entrada para la IFFT (X_m) a partir de un
    vector de K símbolos de datos (a_k), siguiendo las Ecuaciones (2.19)-(2.21).

    Args:
        ak_symbols (np.ndarray): Array de K símbolos.
        dtype (type, optional): Tipo de dato del array de salida. Default: complex.

    Returns:
        np.ndarray: Array de N elementos (X_m), listo para la IFFT.
    """
    assert len(ak_symbols) == p.K, f"Se esperaban {p.K} símbolos, pero se recibieron {len(ak_symbols)}."
    
    if not hasattr(map_symbols_to_ifft_input, 'g_k_params'):
        map_symbols_to_ifft_input.g_k_params = get_baseband_freq_indices()
    
    g_k, g0, gK_minus_1 = map_symbols_to_ifft_input.g_k_params

    # Usamos el dtype proporcionado. Para la prueba simbólica será 'object' o 'str'.
    X_m = np.zeros(p.N, dtype=dtype) 
    
    # Regla 1: DC y Frecuencias Positivas
    m_range_pos = np.arange(0, gK_minus_1 + 1)
    k_indices_pos = m_range_pos - g0
    X_m[m_range_pos] = ak_symbols[k_indices_pos]
    
    # Regla 3: Frecuencias Negativas
    if g0 < 0:
        m_range_neg = np.arange(g0 + p.N, p.N)
        k_indices_neg = m_range_neg - (g0 + p.N)
        X_m[m_range_neg] = ak_symbols[k_indices_neg]
        
    return X_m

print("Módulo 'mapping.py' cargado y listo.")