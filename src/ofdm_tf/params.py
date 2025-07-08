# src/ofdm_tf/params.py

"""
Módulo de Configuración Global para la Simulación OFDM.
Contiene todos los parámetros base y derivados del sistema.
"""

import numpy as np

# --- Generador de Números Aleatorios (para reproducibilidad) ---------
RNG = np.random.default_rng(seed=42)

# --- Parámetros Base de la Simulación ---------------------------------
N = 64          # Tamaño de la FFT (número de subportadoras)
M = 4           # Orden de modulación (4-QPSK)
L_cp = N // 4   # Longitud del prefijo cíclico (25% de N)
N_sym = 10_000  # Número de símbolos OFDM a simular

# --- Parámetros de la Señal ------------------------------------------
Fs = 1.0        # Frecuencia de muestreo (normalizada)
Es = 1.0        # Energía promedio por símbolo de constelación (normalizada)

# --- Parámetros Derivados (calculados automáticamente) ----------------
mu = int(np.log2(M))  # Bits por símbolo de constelación (QPSK -> 2)
Tu = N / Fs           # Duración del símbolo útil (tiempo de la IFFT)
T_cp = L_cp / Fs      # Duración del prefijo cíclico
T_sym = Tu + T_cp     # Duración total del símbolo OFDM
delta_f = 1 / Tu      # Espaciado de frecuencia entre subportadoras

# --- Verificaciones de Coherencia (Sanity Checks) --------------------
assert 2**mu == M, "M debe ser una potencia de 2."
assert N >= L_cp, "El prefijo cíclico no puede ser más largo que el símbolo."
assert (N * mu) % 8 == 0, "Para simplicidad, aseguremos que cada bloque de datos sea un número entero de bytes."

# --- Diccionario de Parámetros (opcional, útil para logging) ---------
PARAMS_DICT = {
    'N': N, 'M': M, 'mu': mu, 'L_cp': L_cp, 'N_sym': N_sym,
    'Fs': Fs, 'Es': Es, 'Tu': Tu, 'T_cp': T_cp, 'T_sym': T_sym,
    'delta_f': delta_f, 'seed': 42
}

print("Módulo de parámetros 'params.py' cargado correctamente.")