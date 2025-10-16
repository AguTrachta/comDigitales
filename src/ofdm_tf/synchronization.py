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

def find_timing_and_freq_offset(rx_signal):
    """
    Encuentra el inicio del paquete y estima el desfase de frecuencia (CFO)
    usando el método de Schmidl & Cox.

    Args:
        rx_signal (np.ndarray): El flujo de muestras complejas recibidas del SDR.

    Returns:
        tuple: (
            best_timing_offset (int): El índice de la muestra donde se estima
                                      que comienza el símbolo de entrenamiento.
            cfo_est_rad_per_sample (float): La estimación del CFO en radianes por muestra.
        )
    """
    table_long = p.N // 2  # Longitud de la mitad del símbolo
    
    # --- Búsqueda del Mejor Punto de Sincronización (Timing) ---
    best_offset = -1
    max_metric = -1
    
    P_d_values = [] # Guardaremos los valores de P(d) para el cálculo del CFO
    
    # El bucle desliza una ventana sobre la señal recibida
    # Dejamos margen al final para que quepa la ventana completa de 2*L
    for d in range(len(rx_signal) - 2*table_long):
        
        # Extraer las dos mitades de la ventana actual
        first_half = rx_signal[d : d+table_long]
        second_half = rx_signal[d+table_long : d+2*table_long]
        
        # --- Calcular P(d) y R(d) ---
        P_d = np.sum(np.conj(first_half) * second_half)
        R_d = np.sum(np.abs(second_half)**2)
        
        P_d_values.append(P_d) # Guardar para después
        
        # --- Calcular la métrica M(d) ---
        if R_d > 0: # Evitar división por cero
            M_d = (np.abs(P_d)**2) / (R_d**2)
        else:
            M_d = 0
            
        # Actualizar si encontramos un pico más alto
        if M_d > max_metric:
            max_metric = M_d
            best_offset = d
            
    # --- Estimación del Desfase de Frecuencia (CFO) ---
    
    # Una vez encontrado el mejor offset, recuperamos el P(d) correspondiente
    P_at_best_offset = P_d_values[best_offset]
    
    # Fórmula (38): El ángulo de P(d) es el desfase de fase total en L muestras
    angle_phi = np.angle(P_at_best_offset)
    
    # Convertir el ángulo total al desfase por muestra (en radianes/muestra)
    # Esto es equivalente a la Fórmula (39) pero en unidades más convenientes
    cfo_est_rad_per_sample = angle_phi / table_long
    
    return best_offset, cfo_est_rad_per_sample