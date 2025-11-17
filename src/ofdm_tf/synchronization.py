import numpy as np
from . import params as p

# ============================================================
#        GENERACIÓN DEL SÍMBOLO DE ENTRENAMIENTO (PREÁMBULO)
# ============================================================

def generate_schmidl_cox_preamble():
    """
    Genera el símbolo de entrenamiento en el dominio del TIEMPO
    según el método de Schmidl & Cox.

    El símbolo tiene la propiedad de que su primera mitad es idéntica
    a su segunda mitad.

    Returns:
        np.ndarray: Un array de N muestras complejas que representan
                    el preámbulo en el dominio del tiempo.
    """
    # 1. Crear el símbolo en el dominio de la FRECUENCIA
    preamble_freq = np.zeros(p.N, dtype=complex)
    
    # 2. Tomar los índices de las subportadoras pares
    # Nos aseguramos de incluir el 0 y excluir N/2 si N es par,
    # que es lo típico para la simetría de la IFFT.
    even_carriers = np.arange(0, p.N, 2)
    
    # 3. Generar una secuencia Pseudo-Aleatoria (PN) para las portadoras pares
    # Usamos QPSK para la secuencia PN. Podría ser cualquier cosa, pero QPSK es simple.
    # Generamos N/2 valores aleatorios de +1 o -1 para las partes real e imaginaria.
    pn_sequence_real = 2 * p.RNG.integers(0, 2, size=len(even_carriers)) - 1
    pn_sequence_imag = 2 * p.RNG.integers(0, 2, size=len(even_carriers)) - 1
    
    pn_qpsk = (pn_sequence_real + 1j * pn_sequence_imag) / np.sqrt(2)
    
    # 4. Asignar la secuencia PN a las portadoras pares
    preamble_freq[even_carriers] = pn_qpsk
    
    # 5. Convertir a dominio del tiempo usando la IFFT
    # La propiedad de "solo frecuencias pares" crea la repetición en el tiempo.
    preamble_time = np.fft.ifft(preamble_freq, norm='ortho')
    
    return preamble_time

# ============================================================
#        ALGORITMOS DE SINCRONIZACIÓN EN EL RECEPTOR
# ============================================================

def calculate_timing_metric(rx_signal):
    """
    Calcula las métricas P(d) y M(d) de Schmidl & Cox para una señal.
    
    Returns:
        P_d_values (list): Lista de los valores complejos de correlación P(d).
        M_d_values (list): Lista de los valores de la métrica de timing M(d).
    """
    L = p.N // 2
    P_d_values = []
    M_d_values = []
    
    for d in range(len(rx_signal) - 2*L):
        first_half = rx_signal[d : d+L]
        second_half = rx_signal[d+L : d+2*L]
        
        P_d = np.sum(np.conj(first_half) * second_half)
        R_d = np.sum(np.abs(second_half)**2)
        
        P_d_values.append(P_d)
        M_d_values.append((np.abs(P_d)**2) / (R_d**2) if R_d > 0 else 0)
        
    return P_d_values, M_d_values

def estimate_timing_offset(M_d_values):
    """
    Encuentra el mejor offset de tiempo a partir de la métrica M(d).
    """
    return np.argmax(M_d_values)

def estimate_cfo(P_d_values, timing_offset):
    """
    Estima el Carrier Frequency Offset (CFO) a partir de P(d) en el
    punto de sincronización.
    """
    L = p.N // 2
    P_at_best_offset = P_d_values[timing_offset]
    angle_phi = np.angle(P_at_best_offset)
    cfo_rad_per_sample = angle_phi / L
    return cfo_rad_per_sample