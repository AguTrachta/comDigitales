"""
Módulo de Bloques de Procesamiento OFDM.

Contiene funciones modulares para cada paso de la cadena de transmisión OFDM, permitiendo su reutilización en diferentes contextos
(tutorial, simulación, implementación).
"""
import numpy as np
from . import params as p
from . import mapping as mp

def generate_bits(num_ofdm_symbols):
    """Bloque 1: Genera una secuencia de bits aleatorios."""
    n_total_bits = num_ofdm_symbols * p.K * p.mu
    return p.RNG.integers(low=0, high=2, size=n_total_bits)

def map_bits_to_symbols(bits_tx):
    """Bloque 2: Mapea bits a símbolos de constelación QPSK."""
    # Paso 1: Reformatear la secuencia de bits en grupos de 'mu' bits.
    # La dimensión resultante será (N_total_de_simbolos, mu).
    bits_reshaped = bits_tx.reshape(-1, p.mu)
    ak_symbols_flat = (1 - 2*bits_reshaped[:, 1]) + 1j*(1 - 2*bits_reshaped[:, 0])
    ak_symbols_flat /= np.sqrt(2)
    return ak_symbols_flat

def build_ifft_input_matrix(ak_symbols_flat, num_ofdm_symbols):
    """Bloque 3: Construye la matriz de entrada para la IFFT."""
    ak_matrix = ak_symbols_flat.reshape(num_ofdm_symbols, p.K)
    X_matrix = np.zeros((num_ofdm_symbols, p.N), dtype=complex)
    for i in range(num_ofdm_symbols):
        X_matrix[i, :] = mp.map_symbols_to_ifft_input(ak_matrix[i, :])
    return X_matrix

def modulate_with_ifft(X_matrix):
    """Bloque 4: Aplica la IFFT para modular."""
    # Usamos np.fft.ifft.
    # - El primer argumento es la matriz de entrada.
    # - axis=1 le dice a NumPy que aplique la IFFT a lo largo de cada fila.
    # - norm='ortho' usa una normalización de 1/sqrt(N) que conserva la energía,
    #   lo cual es conveniente para las verificaciones
    return np.fft.ifft(X_matrix, axis=1, norm='ortho')

def add_cyclic_prefix(x_time):
    """Bloque 5: Añade el prefijo cíclico."""
    # Seleccionar la porción de cada símbolo que será el prefijo.
    # Usamos el rebanado (slicing) de NumPy.
    # x_time[:, -p.L:] selecciona las últimas 'L' columnas de TODAS las filas.
    cyclic_prefix = x_time[:, -p.L:]
    return np.concatenate([cyclic_prefix, x_time], axis=1)

def parallel_to_serial(x_time_with_cp):
    """Bloque 6: Convierte la matriz de símbolos a una trama serie."""
    # La función .flatten() de NumPy
    # "Aplana" una matriz multidimensional en un único vector de 1D,
    # recorriendo las filas de izquierda a derecha y de arriba a abajo.
    return x_time_with_cp.flatten()

def build_ifft_input_matrix_with_pilots(data_symbols_flat, num_ofdm_symbols):
    """
    Construye la matriz de entrada para la IFFT, insertando pilotos
    y los símbolos de datos proporcionados.
    """
    carriers_indices = np.arange(p.K)
    pilot_indices = carriers_indices[::p.PILOT_SPACING]
    data_indices = np.delete(carriers_indices, pilot_indices)
    num_data_per_sym = len(data_indices)

    ak_matrix_with_pilots = np.zeros((num_ofdm_symbols, p.K), dtype=complex)
    ak_matrix_with_pilots[:, pilot_indices] = p.PILOT_VALUE
    
    data_symbols_reshaped = data_symbols_flat.reshape(num_ofdm_symbols, num_data_per_sym)
    ak_matrix_with_pilots[:, data_indices] = data_symbols_reshaped

    X_matrix = np.zeros((num_ofdm_symbols, p.N), dtype=complex)
    for i in range(num_ofdm_symbols):
        X_matrix[i, :] = mp.map_symbols_to_ifft_input(ak_matrix_with_pilots[i, :])
        
    return X_matrix