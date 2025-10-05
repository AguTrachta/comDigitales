# src/ofdm_tf/channel.py

import numpy as np
from . import params as p
from . import utils as u

# --- Definición del Canal Multipath ---
# channel_taps seria h[n] Lindell la define en la Ecuación (5.3) como una suma de deltas de Dirac. 
# Nuestro array es la versión muestreada de esa función, h[n].
CHANNEL_TAPS = np.array([1, # h[0]   → camino directo (retardo 0)
                         0, # h[1]       → no hay eco a 1 muestra
                         np.sqrt(0.5) * np.exp(1j*np.pi/4), # h[2]   → eco a 2 muestras, -3 dB y +45°
                         0, 0, # h[3], h[4] → sin eco a 3 y 4 muestras
                         0.2,
                         0.3,
                         0.4
                         ], dtype=complex) # h[5]   → eco a 5 muestras, -14 dB aprox., fase 0

# --- Parámetros del Canal de Referencia (Tabla 4.2) ---
# Tap | Delay (µs) | Avg. Power (dB)
CHANNEL_TABLE = np.array([
    [1, 0.000, -5.7], [2, 0.217, -7.6], [3, 0.512, -10.1], [4, 0.514, -10.2],
    [5, 0.517, -10.2], [6, 0.674, -11.5], [7, 0.882, -13.4], [8, 1.230, -16.3],
    [9, 1.287, -16.9], [10, 1.311, -17.1], [11, 1.349, -17.4], [12, 1.533, -19.0],
    [13, 1.535, -19.0], [14, 1.622, -19.8], [15, 1.818, -21.5], [16, 1.836, -21.6],
    [17, 1.884, -22.1], [18, 1.943, -22.6], [19, 2.048, -23.5], [20, 2.140, -24.3]
])

def map_channel_to_taps(channel_table, Ts_us=1.0):
    """
    Discretiza el canal de trayectos múltiples (tabla) a la respuesta
    al impulso discreta h[n] usando el periodo de muestreo (Ts_us).
    """
    delays_us = channel_table[:, 1]
    power_db = channel_table[:, 2]

    # 1. Convertir potencias de dB a lineal
    power_lin = 10**(power_db / 10.0)
    
    # 2. Normalizar las potencias para que su suma sea 1
    power_lin_normalized = power_lin / np.sum(power_lin)
    
    # 3. Discretizar los retardos a índices de muestra
    sample_indices = np.round(delays_us / Ts_us).astype(int)
    max_index = np.max(sample_indices) if sample_indices.size > 0 else 0
    
    h_taps = np.zeros(max_index + 1, dtype=complex)
    
    # 4. Generar cada tap con una fase aleatoria
    for n_idx, P_norm in zip(sample_indices, power_lin_normalized):
        # Generar una variable compleja Gaussiana con media 0 y varianza 1
        random_fade = (p.RNG.standard_normal() + 1j * p.RNG.standard_normal()) / np.sqrt(2)
        
        # Asignar la amplitud promedio (sqrt(Potencia)) multiplicada por el desvanecimiento aleatorio
        # Se suma por si múltiples ecos caen en el mismo índice de muestra
        h_taps[n_idx] += np.sqrt(P_norm) * random_fade
        
    return h_taps

# función para obtener el perfil de retardos estático

def get_static_channel_profile(channel_table, Ts_us):
    """
    Calcula la respuesta al impulso que solo contiene la estructura
    de retardos, para determinar la longitud máxima del canal.
    """
    delays_us = channel_table[:, 1]
    sample_indices = np.round(delays_us / Ts_us).astype(int)
    max_index = np.max(sample_indices) if sample_indices.size > 0 else 0
    
    # Creamos un vector que solo tiene '1' en las posiciones de los ecos
    h_profile = np.zeros(max_index + 1)
    h_profile[sample_indices] = 1
    return h_profile

# channel_support calcula la version discreta de la duracion de la respuesta al impulso Tch.
# se fija donde esta el ultimo eco no despreciable

def channel_support(h, mag_rel_thresh=1e-3, energy_frac=None):
    h = np.asarray(h)
    mag2 = np.abs(h)**2
    if energy_frac is not None:
        total = mag2.sum()
        if total == 0:
            return 0, -1
        order = np.argsort(mag2)[::-1]
        mask = np.zeros_like(mag2, dtype=bool)
        acc = 0.0
        for idx in order:
            if mag2[idx] == 0: break
            mask[idx] = True
            acc += mag2[idx]
            if acc >= energy_frac * total:
                break
        nz = np.flatnonzero(mask)
    else:
        thr = mag_rel_thresh * np.max(np.abs(h)) if np.max(np.abs(h))>0 else 0.0
        nz = np.flatnonzero(np.abs(h) > thr)
    if nz.size == 0:
        return 0, -1
    return int(nz[0]), int(nz[-1])

def required_cp_length(h, **kwargs):
    n0, n1 = channel_support(h, **kwargs)
    return max(0, n1 - n0)

# --- Cálculo de la Longitud del Prefijo Cíclico ---
# Esta es la duración de una muestra de la IFFT en microsegundos.
# T_obs está en segundos, fsamp en Hz. Ts = 1/fsamp.
sampling_period_us = (1 / p.fsamp) * 1e6

# CAMBIO 4: Calcular L_CP_req usando el perfil estático
# Creamos el perfil de retardos basado en el Ts real de nuestro sistema.
h_static_profile = get_static_channel_profile(CHANNEL_TABLE, Ts_us=sampling_period_us)

# Ahora calculamos el L_CP_req sobre este perfil estático y predecible.
n0, n1 = channel_support(h_static_profile)
L_h_eff = (n1 - n0 + 1) if n1 >= n0 else 0
L_CP_req = required_cp_length(h_static_profile)

# --- Verificaciones ---
if p.L < L_CP_req:
    raise ValueError(f"CP insuficiente: p.L={p.L} < L_req={L_CP_req} (n0={n0}, n1={n1}).")
if p.L >= p.N:
    raise ValueError("CP no puede ser >= N (no quedaría parte útil).")

"""
Define la función principal que modela el canal de comunicación.
Actúa como un conmutador para seleccionar entre diferentes tipos de canal
(ideal, AWGN, etc.) e invoca las funciones auxiliares correspondientes.
"""

def apply_channel(signal, channel_type="ideal", ebn0_db=None):
    """
    Aplica el modelo de canal seleccionado a la señal de entrada.

    Args:
        signal (np.array): La señal de banda base transmitida (compleja).
        channel_type (str): El tipo de canal a aplicar.
                            Opciones: "ideal", "awgn".
        ebn0_db (float, optional): La relación Eb/N0 en dB. Requerido si
                                   el canal no es "ideal". Defaults to None.

    Returns:
        np.array: La señal después de pasar por el canal.
    """
    if channel_type == "ideal":
        # El canal identidad no altera la señal.
        return signal
    
    elif channel_type == "awgn":
        # Asegurarse de que se proporcionó un valor de Eb/N0.
        if ebn0_db is None:
            raise ValueError("El canal 'awgn' requiere un valor para ebn0_db.")
        
        # Ahora llamamos a la nueva función basada en SNR de utils.
        return u.add_awgn_snr(signal, ebn0_db)
    
    elif channel_type == "multitap_awgn":
        if ebn0_db is None:
            raise ValueError("El canal 'multitap_awgn' requiere un valor para ebn0_db.")
            
        # 1. Generar una NUEVA realización del canal de desvanecimiento
        h = map_channel_to_taps(CHANNEL_TABLE, Ts_us=sampling_period_us) # Usar el Ts real del sistema
        
        # 2. Aplicar la convolución
        signal_multitap = np.convolve(signal, h, mode='same')
        
        # 3. Añadir ruido
        return u.add_awgn_snr(signal_multitap, ebn0_db)

    else:
        raise ValueError(f"Tipo de canal '{channel_type}' no soportado. "
                         "Opciones válidas: 'ideal', 'awgn'.")

print("Módulo 'channel.py' cargado.")
print(f"  - h (taps): {np.round(CHANNEL_TAPS, 3)}")
print(f"  - soporte efectivo: n0={n0}, n1={n1}  => L_h_eff={L_h_eff} taps")
print(f"  - CP requerido: L_req={L_CP_req} (taps). CP configurado: L={p.L}")
print(f"  - Período de muestreo del sistema (Ts): {sampling_period_us:.4f} µs")
