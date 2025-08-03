# src/ofdm_tf/mapping.py

"""
Módulo para el mapeo de bits a símbolos (constelaciones) y
el mapeo de símbolos a subportadoras.
"""
import numpy as np
from . import params as p

def get_subcarrier_indices():
    """
    Calcula los índices de las subportadoras de datos dentro del vector de la IFFT.

    Esta función implementa una estrategia de mapeo de paso bajo, dejando la
    componente DC (índice 0) y la de Nyquist (N/2) a cero.
    Maneja tanto K par como impar.

    Returns:
        (np.ndarray, np.ndarray): Una tupla conteniendo:
            - all_subcarriers: Array con todos los índices de las subportadoras de datos.
            - data_carriers_pos: Array con los índices de las frecuencias positivas.
            - data_carriers_neg: Array con los índices de las frecuencias negativas.
    """
    # Se reserva la componente DC (índice 0)
    # Se crea un espectro simétrico en la medida de lo posible.

    if p.K % 2 == 0:
        # K es par: Mapeo perfectamente simétrico.
        # K/2 para frec. positivas, K/2 para frec. negativas.
        n_pos = p.K // 2
        n_neg = p.K // 2
        
        # Índices para frecuencias positivas (ej: 1, 2, ..., 32)
        carriers_pos = np.arange(1, n_pos + 1)
        # Índices para frecuencias negativas (ej: 80-32, ..., 79)
        carriers_neg = np.arange(p.N - n_neg, p.N)
        
    else:
        # K es impar: Mapeo asimétrico.
        # Se pone una subportadora más en el lado positivo.
        n_pos = (p.K + 1) // 2
        n_neg = (p.K - 1) // 2
        
        carriers_pos = np.arange(1, n_pos + 1)
        carriers_neg = np.arange(p.N - n_neg, p.N)

    all_carriers = np.concatenate([carriers_pos, carriers_neg])
    
    # Verificación de consistencia interna
    assert len(all_carriers) == p.K
    
    return all_carriers, carriers_pos, carriers_neg

# Pre-calculamos los índices para que estén disponibles en el módulo
# Esta es una buena práctica para no recalcularlos cada vez.
ALL_SUBCARRIER_INDICES, DATA_CARRIERS_POS, DATA_CARRIERS_NEG = get_subcarrier_indices()

print("Módulo 'mapping.py' cargado. Índices de subportadoras calculados.")