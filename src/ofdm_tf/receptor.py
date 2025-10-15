"""
Módulo de Bloques de Procesamiento OFDM.

Contiene funciones modulares para cada paso de la cadena de receptor OFDM, permitiendo su reutilización en diferentes contextos
(tutorial, simulación, implementación).
"""
import numpy as np
from . import params as p
from . import demapping as dmp

def serial_to_parallel(rx_signal, num_ofdm_symbols):
    """Bloque 8: Convierte la trama serie recibida en una matriz de símbolos."""
    return rx_signal.reshape(num_ofdm_symbols, p.N + p.L)

def remove_cyclic_prefix(rx_matrix_with_cp):
    """Bloque 9: Elimina el prefijo cíclico."""
    return rx_matrix_with_cp[:, p.L:]

def demodulate_with_fft(rx_matrix_no_cp):
    """Bloque 10: Aplica la FFT para demodular."""
    return np.fft.fft(rx_matrix_no_cp, axis=1, norm='ortho')

def extract_and_demap_symbols(Y_matrix, num_ofdm_symbols):
    """Bloque 11: Extrae los símbolos de datos y los demapea a bits."""
    ak_recovered_matrix = np.zeros((num_ofdm_symbols, p.K), dtype=complex)
    for i in range(num_ofdm_symbols):
        ak_recovered_matrix[i, :] = dmp.extract_symbols_from_fft_output(Y_matrix[i, :])
    
    ak_symbols_rx_flat = ak_recovered_matrix.flatten()
    bits_rx = dmp.demap_symbols_to_bits(ak_symbols_rx_flat)
    return bits_rx