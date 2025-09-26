"""
utils.py
Funciones auxiliares para simulación de sistemas OFDM en canal AWGN.
Incluye conversiones dB <-> lineal, cálculo de varianza de ruido,
y generación de ruido gaussiano complejo.
"""

import time
import numpy as np
from . import params as p
from . import mapping as mp
from . import demapping as dmp
from . import channel as ch
from . import transmisor as transmisor
from . import receptor as receptor

# ============================================================
# Conversión entre escalas
# ============================================================

def db2lin(x_dB):
    """
    Convierte de decibelios a escala lineal.
    """
    return 10**(x_dB/10.0)


def lin2db(x_lin):
    """
    Convierte de escala lineal a decibelios.
    """
    return 10*np.log10(x_lin)


# ============================================================
# Varianza del ruido
# ============================================================

def calculate_noise_variance(EbN0_dB):
    """
    Calcula la varianza del ruido complejo para un valor dado de Eb/N0.
    
    Parámetros:
        EbN0_dB : float
            Relación Eb/N0 en dB.
    
    Retorna:
        N0 : float
            Varianza total del ruido complejo (N0).
            Cada componente (real e imaginaria) tiene varianza N0/2.
    """
    ebn0_lin = db2lin(EbN0_dB)      # Eb/N0 en lineal
    Eb = p.Es / p.mu                # Energía por bit (Es / bits por símbolo)
    N0 = Eb / ebn0_lin              # densidad espectral de ruido
    return N0


def calculate_noise_variance_with_cp(EbN0_dB, eta):
    """
    Calcula la varianza del ruido considerando la penalización por prefijo cíclico.
    
    Parámetros:
        EbN0_dB : float
            Relación Eb/N0 en dB.
        eta : float
            Eficiencia útil del símbolo OFDM (Tu / (Tu+Tcp)).
    
    Retorna:
        N0 : float
            Varianza total del ruido complejo (ajustada por CP).
    """
    ebn0_lin = db2lin(EbN0_dB)
    Eb = p.Es / p.mu
    Eb_eff = eta * Eb               # energía útil reducida por CP
    N0 = Eb_eff / ebn0_lin
    return N0


# ============================================================
# Canal AWGN
# ============================================================

def calculate_snr(EbN0_dB):
    """
    Convierte Eb/N0 a SNR para este sistema OFDM específico.
    La SNR depende de la eficiencia espectral (bits por muestra).
    """
    ebn0_lin = db2lin(EbN0_dB)
    
    # La eficiencia se define por los bits de datos (K*mu) que se envían
    # en el tiempo que ocupan N muestras de la IFFT (la parte útil).
    efficiency = (p.K * p.mu) / p.N
    
    snr_lin = ebn0_lin * efficiency
    return snr_lin

def add_awgn_snr(signal, ebn0_db):
    """
    Agrega ruido AWGN a una señal compleja basado en la SNR.
    Este es el método correcto para un modelo de POTENCIA CONSTANTE.
    
    Args:
        signal (np.ndarray): Señal baseband compleja (serializada).
        ebn0_db (float): Relación Eb/N0 en dB.
    
    Returns:
        np.ndarray: Señal con ruido agregado.
    """
    # 1. Calcular la potencia promedio real de la señal transmitida
    signal_power = np.mean(np.abs(signal)**2)
    
    # 2. Convertir el Eb/N0 deseado a la SNR correspondiente para este sistema
    snr_lin = calculate_snr(ebn0_db)
    
    # 3. Calcular la potencia de ruido necesaria para alcanzar esa SNR
    # SNR = Potencia_Señal / Potencia_Ruido
    # => Potencia_Ruido = Potencia_Señal / SNR
    noise_power = signal_power / snr_lin
    
    # 4. Generar ruido con esa potencia.
    # La potencia (varianza) se reparte entre la parte real y la imaginaria.
    sigma = np.sqrt(noise_power / 2)
    noise = sigma * (p.RNG.standard_normal(*signal.shape) + 1j * p.RNG.standard_normal(*signal.shape))
    
    return signal + noise

def run_montecarlo_simulation(ebn0_db_range, min_errors, max_bits):
    """
    Ejecuta una simulación OFDM de Monte Carlo sobre un rango de Eb/N0.
    (La descripción de la función sigue siendo la misma)
    """
    print("--- Iniciando Simulación de Monte Carlo para Curva BER ---")
    ber_results = []
    start_time_total = time.time()

    for ebn0_db in ebn0_db_range:
        total_bits_simulados = 0
        total_errores = 0
        start_time_point = time.time()
        
        print(f"\nSimulando para Eb/N0 = {ebn0_db} dB...")

        while total_errores < min_errors and total_bits_simulados < max_bits:

            # --- Transmisor ---
            bits_tx = transmisor.generate_bits(p.N_sym)
            ak_symbols = transmisor.map_bits_to_symbols(bits_tx)
            X_matrix = transmisor.build_ifft_input_matrix(ak_symbols, p.N_sym)
            x_time = transmisor.modulate_with_ifft(X_matrix)
            x_time_with_cp = transmisor.add_cyclic_prefix(x_time)
            tx_signal = transmisor.parallel_to_serial(x_time_with_cp)
            
            # --- Canal ---
            rx_signal = ch.apply_channel(tx_signal, "awgn", ebn0_db)
            
            # --- Receptor ---
            rx_matrix_with_cp = receptor.serial_to_parallel(rx_signal, p.N_sym)
            rx_matrix_no_cp = receptor.remove_cyclic_prefix(rx_matrix_with_cp)
            Y_matrix = receptor.demodulate_with_fft(rx_matrix_no_cp)
            bits_rx = receptor.extract_and_demap_symbols(Y_matrix, p.N_sym)
            
            # --- Acumular resultados ---
            errores_en_tanda = np.sum(bits_tx != bits_rx)
            total_errores += errores_en_tanda
            total_bits_simulados += len(bits_tx)
            
            print(f"\r  -> Bits simulados: {total_bits_simulados}, Errores contados: {total_errores}", end="")

        # --- 6. Calcular y guardar el BER para este punto ---
        ber_calculado = total_errores / total_bits_simulados if total_bits_simulados > 0 else 0
        ber_results.append(ber_calculado)
        
        end_time_point = time.time()
        print(f"\n  -> BER final: {ber_calculado:.3e} (calculado en {end_time_point - start_time_point:.2f}s)")
        
        if ber_calculado == 0 and total_bits_simulados >= max_bits:
            print("  Límite de bits alcanzado con 0 errores. Rellenando el resto de los puntos.")
            while len(ber_results) < len(ebn0_db_range):
                ber_results.append(0)
            break

    end_time_total = time.time()
    print(f"\n--- Simulación de Monte Carlo completada en {end_time_total - start_time_total:.2f} segundos ---")
    return ber_results