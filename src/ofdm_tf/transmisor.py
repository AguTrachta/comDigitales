"""
Módulo de Bloques de Procesamiento OFDM.

Contiene funciones modulares para cada paso de la cadena de transmisión OFDM, permitiendo su reutilización en diferentes contextos
(tutorial, simulación, implementación).
"""
import numpy as np
from . import params as p
from . import mapping as mp
from . import synchronization as sync

def generate_all_bits(num_ofdm_symbols):
    """Genera bits para llenar TODAS las K_TOTAL subportadoras."""
    # Esta función asume un escenario sin pilotos
    n_total_bits = num_ofdm_symbols * p.K_TOTAL * p.mu
    return p.RNG.integers(low=0, high=2, size=n_total_bits)

def generate_data_bits(num_ofdm_symbols):
    """Genera bits para llenar solo las K_DATA subportadoras."""
    # Usa el parámetro K_DATA calculado en params.py
    n_data_bits = num_ofdm_symbols * p.K_DATA * p.mu
    return p.RNG.integers(low=0, high=2, size=n_data_bits)

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

# En transmisor.py

def build_full_frame(bits_tx, num_data_symbols):
    """
    Construye una trama OFDM completa, incluyendo el preámbulo de
    sincronización y los símbolos de datos.
    """
    # 1. Generar el preámbulo de Schmidl & Cox
    preamble_time = sync.generate_schmidl_cox_preamble()
    
    # 2. Añadirle su propio Prefijo Cíclico
    preamble_cp = preamble_time[-p.L:]
    preamble_with_cp = np.concatenate([preamble_cp, preamble_time])
    
    # 3. Procesar los bits de datos
    ak_symbols = map_bits_to_symbols(bits_tx)
    
    # Decidir si usar la función con o sin pilotos
    if p.PILOT_SPACING is not None and p.PILOT_SPACING > 0:
        X_matrix = build_ifft_input_matrix_with_pilots(ak_symbols, num_data_symbols)
    else:
        X_matrix = build_ifft_input_matrix(ak_symbols, num_data_symbols)

    x_time = modulate_with_ifft(X_matrix)
    x_time_with_cp = add_cyclic_prefix(x_time)
    data_payload = parallel_to_serial(x_time_with_cp)
    
    # 4. Unir el preámbulo y los datos para formar la trama final
    full_frame = np.concatenate([preamble_with_cp, data_payload])
    
    return full_frame