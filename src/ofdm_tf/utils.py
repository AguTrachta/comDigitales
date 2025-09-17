"""
utils.py
Funciones auxiliares para simulación de sistemas OFDM en canal AWGN.
Incluye conversiones dB <-> lineal, cálculo de varianza de ruido,
y generación de ruido gaussiano complejo.
"""

import numpy as np


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

def add_awgn(signal, EbN0_dB):
    """
    Agrega ruido AWGN a una señal compleja.
    
    Parámetros:
        signal : ndarray
            Señal baseband compleja (puede ser OFDM ya modulado).
        EbN0_dB : float
            Relación Eb/N0 en dB.
    
    Retorna:
        signal_noisy : ndarray
            Señal con ruido agregado.
    """
    N0 = calculate_noise_variance(EbN0_dB)
    sigma = np.sqrt(N0/2)   # desviación estándar por dimensión
    noise = sigma * (np.random.randn(*signal.shape) + 1j*np.random.randn(*signal.shape))
    return signal + noise


def add_awgn_with_cp(signal, EbN0_dB, eta):
    """
    Agrega ruido AWGN a una señal considerando la penalización del CP.
    
    Parámetros:
        signal : ndarray
            Señal baseband compleja.
        EbN0_dB : float
            Relación Eb/N0 en dB.
        eta : float
            Eficiencia útil (Tu / (Tu+Tcp)).
    
    Retorna:
        signal_noisy : ndarray
            Señal con ruido agregado (ajustada por CP).
    """
    N0 = calculate_noise_variance_with_cp(EbN0_dB, eta)
    sigma = np.sqrt(N0/2)
    noise = sigma * (np.random.randn(*signal.shape) + 1j*np.random.randn(*signal.shape))
    return signal + noise
