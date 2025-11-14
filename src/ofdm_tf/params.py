# src/ofdm_tf/params.py

"""
Módulo de Configuración Global para la Simulación OFDM.
Contiene todos los parámetros base y derivados del sistema.
"""

import numpy as np

# --- Generador de Números Aleatorios (para reproducibilidad) ---------
RNG = np.random.default_rng(seed=42)

# --- Parámetros Base de la Simulación ---------------------------------
K_TOTAL = 12          # número de subportadoras
N = 16          # Tamaño de la IFFT/DFT (N > K). Cantidad de muestras.
                # Elegimos una potencia de 2 o un número con factores pequeños para una IFFT eficiente.
M = 4           # Orden de modulación (4-QPSK)
L = 2   # Longitud del prefijo cíclico
N_sym = 7  # Número de símbolos OFDM a simular

# --- Parámetros de la Señal ------------------------------------------
Es = 1.0        # Energía promedio por símbolo de constelación (normalizada)
f_delta = 1.0   # Espaciado entre subportadoras (normalizado)

# --- Parámetros Derivados (calculados automáticamente) ----------------
mu = int(np.log2(M))  # Bits por símbolo (log2(4) = 2 para QPSK)
T_obs = 1 / f_delta   # Duración del símbolo útil (sin CP)
fsamp = N * f_delta   # Frecuencia de muestreo
T_cp = L / fsamp      # Duración del prefijo cíclico
T_s = T_obs + T_cp    # Duración total del símbolo OFDM

# --- Parámetros de Pilotos ---
PILOT_SPACING = 4  # Inserta un piloto cada 4 subportadoras
PILOT_VALUE = (1+1j) / np.sqrt(2) # El valor complejo que tendrán los pilotos (conocido por Tx y Rx)

# Calcular índices de portadoras de datos y pilotos
all_carriers_indices = np.arange(K_TOTAL)
pilot_indices = all_carriers_indices[::PILOT_SPACING]
data_indices = np.delete(all_carriers_indices, pilot_indices)

K_DATA = len(data_indices)      # Número de subportadoras de DATOS
K_PILOTS = len(pilot_indices)   # Número de subportadoras de PILOTOS

# Renombramos K a K_TOTAL para mayor claridad en el resto del código
K = K_TOTAL

# --- Verificaciones de Coherencia (Sanity Checks) --------------------
print(f"Número de subportadoras de datos (K): {K}")
print(f"Tamaño de la IFFT (N): {N}")
print(f"Longitud del Prefijo Cíclico (L): {L}")
print(f"Duración útil (T_obs): {T_obs:.2f} s")
print(f"Duración CP (T_cp): {T_cp:.2f} s")
print(f"Duración total del símbolo (T_s): {T_s:.2f} s")
print(f"Frecuencia de muestreo (fsamp): {fsamp:.2f} Hz")
print(f"Espaciado de subportadoras (f_delta): {f_delta:.2f} Hz")

assert N >= K_TOTAL, "El tamaño de la IFFT (N) debe ser mayor o igual que el número de subportadoras (K)."
print("\n--- Parámetros configurados correctamente ---")