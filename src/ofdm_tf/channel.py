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
                         0.2], dtype=complex) # h[5]   → eco a 5 muestras, -14 dB aprox., fase 0

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
n0, n1 = channel_support(CHANNEL_TAPS, mag_rel_thresh=1e-3)  # o energy_frac=0.99
L_h_eff = (n1 - n0 + 1) if n1 >= n0 else 0
L_CP_req = required_cp_length(CHANNEL_TAPS, mag_rel_thresh=1e-3)

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
        
        # Calcular la penalización por el prefijo cíclico (eta)
        # eta = Tiempo útil / Tiempo total
        eta = p.N / (p.N + p.L)
        
        # Invocar la función de utils que ya considera la penalización del CP.
        return u.add_awgn_with_cp(signal, ebn0_db, eta)
        
    # Aquí se podrían añadir más tipos de canal en el futuro (ej. "rayleigh")
    # elif channel_type == "rayleigh":
    #     ...
        
    else:
        raise ValueError(f"Tipo de canal '{channel_type}' no soportado. "
                         "Opciones válidas: 'ideal', 'awgn'.")

print("Módulo 'channel.py' cargado.")
print(f"  - h (taps): {np.round(CHANNEL_TAPS, 3)}")
print(f"  - soporte efectivo: n0={n0}, n1={n1}  => L_h_eff={L_h_eff} taps")
print(f"  - CP requerido: L_req={L_CP_req} (taps). CP configurado: L={p.L}")
