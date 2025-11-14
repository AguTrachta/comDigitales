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
from . import synchronization as sync

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

def run_montecarlo_simulation(ebn0_db_range, min_errors, max_bits, channel_type="awgn"):
    """
    Ejecuta una simulación OFDM de Monte Carlo sobre un rango de Eb/N0.
    Utiliza los parámetros globales (p.N_sym) para definir el tamaño de cada tanda.
    """
    print(f"--- Iniciando Simulación de Monte Carlo para Canal '{channel_type}' ---")
    ber_results = []
    start_time_total = time.time()

    # Determinar si este modo de simulación necesita pilotos y ecualizador
    use_pilots_and_eq = (channel_type == "multitap_eq")

    for ebn0_db in ebn0_db_range:
        total_bits_simulados = 0
        total_errores = 0
        start_time_point = time.time()
        
        print(f"\nSimulando para Eb/N0 = {ebn0_db} dB...")

        while total_errores < min_errors and total_bits_simulados < max_bits:
            # --- Transmisor ---
            if use_pilots_and_eq:
                # 1. Calcular cuántos bits de DATOS se necesitan por tanda
                carriers = np.arange(p.K)
                pilot_idx = carriers[::p.PILOT_SPACING]
                data_idx  = np.delete(carriers, pilot_idx)
                num_data_per_sym = len(data_idx)
                num_data_bits_per_batch = p.N_sym * num_data_per_sym * p.mu
                
                # 2. Generar EXACTAMENTE ese número de bits aleatorios
                #    En lugar de usar la función antigua, lo hacemos directamente.
                bits_tx_to_compare = p.RNG.integers(low=0, high=2, size=num_data_bits_per_batch)

                # 3. [PASO QUE FALTABA] Mapear esos bits a símbolos QPSK complejos
                data_symbols_flat = transmisor.map_bits_to_symbols(bits_tx_to_compare)

                # 4. Ahora sí, construir la matriz de IFFT con los símbolos de datos correctos
                X_matrix = transmisor.build_ifft_input_matrix_with_pilots(data_symbols_flat, p.N_sym)
            else:
                # Generar un bloque completo de bits (todos son datos)
                bits_tx_to_compare = transmisor.generate_all_bits(p.N_sym)
                ak_symbols = transmisor.map_bits_to_symbols(bits_tx_to_compare)
                X_matrix = transmisor.build_ifft_input_matrix(ak_symbols, p.N_sym)

            # Resto de la cadena del transmisor (común a ambos casos)
            x_time = transmisor.modulate_with_ifft(X_matrix)
            x_time_with_cp = transmisor.add_cyclic_prefix(x_time)
            data_payload_tx = transmisor.parallel_to_serial(x_time_with_cp)

            # --- NUEVO: Construir la Trama Completa ---
            # Ahora añadimos el preámbulo a la carga útil de datos.
            tx_full_frame = transmisor.build_full_frame(data_payload_tx)

            # --- Canal ---
            # Aplicamos el canal a la trama completa (preámbulo + datos).
            # (No se añade padding, el receptor debe encontrar el inicio en la muestra 0)
            rx_full_frame = ch.apply_channel(tx_full_frame, channel_type, ebn0_db)
            
            # --- Receptor ---
            
            # --- NUEVO: Sincronización de Trama ---
            _, M_d_values = sync.calculate_timing_metric(rx_full_frame)
            
            # Usamos el método robusto. Con ruido, puede que no supere 0.95,
            # así que usamos argmax como respaldo si falla.
            umbral_deteccion = 0.9 # Bajamos un poco el umbral para el ruido
            try:
                best_offset = np.nonzero(np.array(M_d_values) > umbral_deteccion)[0][0]
            except IndexError:
                best_offset = sync.estimate_timing_offset(M_d_values)
                
            # Extraer la carga útil basándose en el offset detectado
            len_preambulo_cp = p.L + p.N
            start_of_data = best_offset + len_preambulo_cp
            len_data_payload = p.N_sym * (p.N + p.L)
            end_of_data = start_of_data + len_data_payload
            
            rx_data_payload = rx_full_frame[start_of_data:end_of_data]
            
            # --- Verificación de Sincronización ---
            # Si la sincronización falló, la longitud será incorrecta.
            # En ese caso, contamos todos los bits de la tanda como errores.
            if rx_data_payload.shape[0] != len_data_payload:
                total_errores += len(bits_tx_to_compare)
                total_bits_simulados += len(bits_tx_to_compare)
                print(f"\r  -> ¡Fallo de Sincronización! Bits: {total_bits_simulados}, Errores: {total_errores}", end="")
                continue # Pasar a la siguiente iteración de la tanda
                
            # --- Continuación de la cadena del receptor con los datos sincronizados ---
            rx_matrix_with_cp = receptor.serial_to_parallel(rx_data_payload, p.N_sym)
            rx_matrix_no_cp = receptor.remove_cyclic_prefix(rx_matrix_with_cp)
            Y_matrix = receptor.demodulate_with_fft(rx_matrix_no_cp)
            
            if use_pilots_and_eq:
                X_equalized_symbols = receptor.estimate_and_equalize(Y_matrix)
                bits_rx = receptor.demap_equalized_symbols(X_equalized_symbols)
            else:
                bits_rx = receptor.extract_and_demap_symbols(Y_matrix, p.N_sym)

            # --- Acumular Resultados ---
            errores_en_tanda = np.sum(bits_tx_to_compare != bits_rx)
            total_errores += errores_en_tanda
            total_bits_simulados += len(bits_tx_to_compare)
            
            print(f"\r  -> Bits simulados: {total_bits_simulados}, Errores contados: {total_errores}", end="")
        # --- Calcular y guardar el BER ---
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