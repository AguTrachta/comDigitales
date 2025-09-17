# src/ofdm_tf/demapping.py

"""
Módulo para el demapeo de subportadoras a símbolos y
el demapeo de símbolos a bits.
"""
import numpy as np
from . import params as p
from . import mapping as mp # Importamos mapping para reusar get_baseband_freq_indices

def extract_symbols_from_fft_output(Y_m_vector):
    """
    Extrae los K símbolos de datos (a_k) de un vector de N puntos de la FFT (Y_m).
    Esta función es la inversa de 'map_symbols_to_ifft_input'.

    Args:
        Y_m_vector (np.ndarray): Array de N puntos complejos de la salida de la FFT.

    Returns:
        np.ndarray: Array de K símbolos complejos recuperados.
    """
    assert len(Y_m_vector) == p.N, f"Se esperaban {p.N} puntos, pero se recibieron {len(Y_m_vector)}."
    
    # Obtenemos los mismos parámetros g_k que se usaron en el mapeo
    g_k, g0, gK_minus_1 = mp.get_baseband_freq_indices()
    
    ak_recovered = np.zeros(p.K, dtype=complex)
    
    # Invertimos la Regla 1: Extraer datos de DC y Frecuencias Positivas
    m_range_pos = np.arange(0, gK_minus_1 + 1)
    k_indices_pos = m_range_pos - g0
    ak_recovered[k_indices_pos] = Y_m_vector[m_range_pos]
    
    # Invertimos la Regla 3: Extraer datos de Frecuencias Negativas
    if g0 < 0:
        m_range_neg = np.arange(g0 + p.N, p.N)
        k_indices_neg = m_range_neg - (g0 + p.N)
        ak_recovered[k_indices_neg] = Y_m_vector[m_range_neg]
        
    return ak_recovered

def demap_symbols_to_bits(ak_symbols):
    """
    Convierte (demapea) un array de símbolos complejos QPSK a un array de bits.
    Utiliza una detección de mínima distancia (hard decision).

    Args:
        ak_symbols (np.ndarray): Array de K símbolos complejos QPSK.

    Returns:
        np.ndarray: Array de K*mu bits (0s y 1s).
    """
    # Para QPSK, la decisión se basa en el signo del eje real e imaginario.
    # Esto es equivalente a encontrar el punto de la constelación más cercano.
    
    # Inicializamos una matriz para los bits recuperados
    bits_recovered = np.zeros((len(ak_symbols), p.mu), dtype=int)
    
    # Demapeo de la parte imaginaria para obtener b1 (MSB)
    # Si Im(ak) > 0, el bit original era 0. Si Im(ak) < 0, el bit original era 1.
    bits_recovered[:, 0] = (np.imag(ak_symbols) < 0).astype(int)

    # Demapeo de la parte real para obtener b2 (LSB)
    # Si Re(ak) > 0, el bit original era 0. Si Re(ak) < 0, el bit original era 1.
    bits_recovered[:, 1] = (np.real(ak_symbols) < 0).astype(int)

    # Aplanamos la matriz para obtener la secuencia final de bits
    return bits_recovered.flatten()


print("Módulo 'demapping.py' cargado y listo.")