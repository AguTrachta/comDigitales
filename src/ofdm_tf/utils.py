import numpy as np

def db2lin(db):
    """Decibelios → magnitud lineal"""
    return 10.0 ** (db / 10.0)

def lin2db(x):
    """Magnitud lineal → decibelios"""
    return 10.0 * np.log10(x)

def ebn0_to_noise_variance(EbN0_dB, mu=mu, Es=Es):
    """
    Calcula σ² para ruido AWGN complejo (N0/2 por dimensión) dado Eb/N0 [dB].
    Suponemos señales normalizadas a Es=1.
    """
    EbN0 = db2lin(EbN0_dB)
    return Es / (2 * mu * EbN0)
