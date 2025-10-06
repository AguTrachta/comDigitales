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

def estimate_and_equalize(Y_matrix):
    """
    Realiza la estimación de canal LS, interpolación lineal y ecualización ZF.
    """
    num_ofdm_symbols = Y_matrix.shape[0]
    
    # Extraer los símbolos recibidos (pilotos y datos) de la salida de la FFT
    symbols_rx_matrix = np.zeros((num_ofdm_symbols, p.K), dtype=complex)
    for i in range(num_ofdm_symbols):
        symbols_rx_matrix[i, :] = dmp.extract_symbols_from_fft_output(Y_matrix[i, :])

    # Determinar índices de pilotos y datos
    all_carriers = np.arange(p.K)
    pilot_carriers = all_carriers[::p.PILOT_SPACING]
    data_carriers = np.delete(all_carriers, pilot_carriers)

    # Extraer los valores recibidos en las posiciones de los pilotos
    Y_pilots = symbols_rx_matrix[:, pilot_carriers]
    
    # --- PASO 2: ESTIMACIÓN LS EN PILOTOS ---
    # H_est = Y_pilots / X_pilots
    H_est_at_pilots = Y_pilots / p.PILOT_VALUE

    # --- PASO 3: INTERPOLACIÓN ---
    # Creamos la matriz para guardar el canal estimado completo
    H_est_full = np.zeros((num_ofdm_symbols, p.K), dtype=complex)
    
    # Para cada símbolo OFDM, interpolamos el canal
    for i in range(num_ofdm_symbols):
        # np.interp es la función de interpolación lineal de NumPy
        H_est_full_real = np.interp(all_carriers, pilot_carriers, H_est_at_pilots[i, :].real)
        H_est_full_imag = np.interp(all_carriers, pilot_carriers, H_est_at_pilots[i, :].imag)
        H_est_full[i, :] = H_est_full_real + 1j*H_est_full_imag


    # --- PASO 4: ECUALIZACIÓN ZF ---
    # Extraer los valores recibidos en las posiciones de los datos
    Y_data = symbols_rx_matrix[:, data_carriers]
    
    # Extraer las estimaciones del canal en las posiciones de los datos
    H_est_at_data = H_est_full[:, data_carriers]
    
    # Aplicar el ecualizador Zero-Forcing
    X_eq_data = Y_data / H_est_at_data
    
    return X_eq_data.flatten() # Devolvemos un vector plano de símbolos ecualizados

def demap_equalized_symbols(X_eq_flat):
    """Toma los símbolos ecualizados y los convierte a bits."""
    return dmp.demap_symbols_to_bits(X_eq_flat)