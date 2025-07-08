# src/ofdm_tf/utils.py

"""
Módulo de Funciones Auxiliares para la Simulación OFDM.
Incluye conversiones, cálculo de ruido, etc.
"""
import numpy as np
from . import params as p # Importamos los parámetros para usarlos aquí

def db2lin(db_val):
    """Convierte un valor en decibelios (dB) a una escala lineal."""
    return 10.0**(db_val / 10.0)

def lin2db(lin_val):
    """Convierte un valor en escala lineal a decibelios (dB)."""
    return 10.0 * np.log10(lin_val)

def calculate_noise_variance(EbN0_dB):
    """
    Calcula la varianza del ruido (σ²) para un canal AWGN complejo.

    Args:
        EbN0_dB (float): Relación energía por bit a densidad espectral de ruido, en dB.

    Returns:
        float: Varianza del ruido σ².
    """
    # Eb/N0 en escala lineal
    ebn0_lin = db2lin(EbN0_dB)
    
    # Energía por bit (Eb)
    # Como Es = mu * Eb, entonces Eb = Es / mu
    Eb = p.Es / p.mu
    
    # Densidad espectral de ruido (N0)
    N0 = Eb / ebn0_lin
    
    # La varianza del ruido complejo es N0.
    # (N0/2 para la parte real y N0/2 para la imaginaria)
    return N0

print("Módulo de utilidades 'utils.py' cargado correctamente.")