"""
Módulo de Bloques de Procesamiento OFDM.

Contiene funciones modulares para cada paso de la cadena de transmisión OFDM, permitiendo su reutilización en diferentes contextos
(tutorial, simulación, implementación).
"""
import numpy as np
from . import params as p
from . import mapping as mp
from . import synchronization as sync

def generate_all_bits(num_ofdm_symbols):
    """Genera bits para llenar TODAS las K_TOTAL subportadoras."""
    # Esta función asume un escenario sin pilotos
    n_total_bits = num_ofdm_symbols * p.K_TOTAL * p.mu
    return p.RNG.integers(low=0, high=2, size=n_total_bits)

def generate_data_bits(num_ofdm_symbols):
    """Genera bits para llenar solo las K_DATA subportadoras."""
    # Usa el parámetro K_DATA calculado en params.py
    n_data_bits = num_ofdm_symbols * p.K_DATA * p.mu
    return p.RNG.integers(low=0, high=2, size=n_data_bits)

def map_bits_to_symbols(bits_tx):
    """Bloque 2: Mapea bits a símbolos de constelación QPSK."""
    # Paso 1: Reformatear la secuencia de bits en grupos de 'mu' bits.
    # La dimensión resultante será (N_total_de_simbolos, mu).
    bits_reshaped = bits_tx.reshape(-1, p.mu)
    ak_symbols_flat = (1 - 2*bits_reshaped[:, 1]) + 1j*(1 - 2*bits_reshaped[:, 0])
    ak_symbols_flat /= np.sqrt(2)
    return ak_symbols_flat

def build_ifft_input_matrix(ak_symbols_flat, num_ofdm_symbols):
    """Bloque 3: Construye la matriz de entrada para la IFFT."""
    ak_matrix = ak_symbols_flat.reshape(num_ofdm_symbols, p.K)
    X_matrix = np.zeros((num_ofdm_symbols, p.N), dtype=complex)
    for i in range(num_ofdm_symbols):
        X_matrix[i, :] = mp.map_symbols_to_ifft_input(ak_matrix[i, :])
    return X_matrix

def modulate_with_ifft(X_matrix):
    """Bloque 4: Aplica la IFFT para modular."""
    # Usamos np.fft.ifft.
    # - El primer argumento es la matriz de entrada.
    # - axis=1 le dice a NumPy que aplique la IFFT a lo largo de cada fila.
    # - norm='ortho' usa una normalización de 1/sqrt(N) que conserva la energía,
    #   lo cual es conveniente para las verificaciones
    return np.fft.ifft(X_matrix, axis=1, norm='ortho')

def add_cyclic_prefix(x_time):
    """Bloque 5: Añade el prefijo cíclico."""
    # Seleccionar la porción de cada símbolo que será el prefijo.
    # Usamos el rebanado (slicing) de NumPy.
    # x_time[:, -p.L:] selecciona las últimas 'L' columnas de TODAS las filas.
    cyclic_prefix = x_time[:, -p.L:]
    return np.concatenate([cyclic_prefix, x_time], axis=1)

def parallel_to_serial(x_time_with_cp):
    """Bloque 6: Convierte la matriz de símbolos a una trama serie."""
    # La función .flatten() de NumPy
    # "Aplana" una matriz multidimensional en un único vector de 1D,
    # recorriendo las filas de izquierda a derecha y de arriba a abajo.
    return x_time_with_cp.flatten()

def build_ifft_input_matrix_with_pilots(data_symbols_flat, num_ofdm_symbols):
    """
    Construye la matriz de entrada para la IFFT, insertando pilotos
    y los símbolos de datos proporcionados.
    """
    carriers_indices = np.arange(p.K)
    pilot_indices = carriers_indices[::p.PILOT_SPACING]
    data_indices = np.delete(carriers_indices, pilot_indices)
    num_data_per_sym = len(data_indices)

    ak_matrix_with_pilots = np.zeros((num_ofdm_symbols, p.K), dtype=complex)
    ak_matrix_with_pilots[:, pilot_indices] = p.PILOT_VALUE
    
    data_symbols_reshaped = data_symbols_flat.reshape(num_ofdm_symbols, num_data_per_sym)
    ak_matrix_with_pilots[:, data_indices] = data_symbols_reshaped

    X_matrix = np.zeros((num_ofdm_symbols, p.N), dtype=complex)
    for i in range(num_ofdm_symbols):
        X_matrix[i, :] = mp.map_symbols_to_ifft_input(ak_matrix_with_pilots[i, :])
        
    return X_matrix

def build_full_frame(data_payload):
    """
    Construye una trama OFDM completa anteponiendo el preámbulo de
    sincronización a una carga útil de datos ya procesada.

    Args:
        data_payload (np.ndarray): La señal de datos ya modulada, con CP y
                                   serializada (la salida del Bloque 6).

    Returns:
        np.ndarray: La trama completa [PREÁMBULO_CON_CP, DATOS].
    """
    # 1. Generar el preámbulo de Schmidl & Cox en el dominio del tiempo
    preamble_time = sync.generate_schmidl_cox_preamble()
    
    # 2. Añadirle su propio Prefijo Cíclico
    preamble_cp = preamble_time[-p.L:]
    preamble_with_cp = np.concatenate([preamble_cp, preamble_time])
    
    # 3. Unir el preámbulo y la carga útil de datos
    full_frame = np.concatenate([preamble_with_cp, data_payload])
    
    return full_frame

def create_packet_from_bits(data_bits, use_pilots=False):
    """
    Función maestra del transmisor: crea un paquete OFDM completo.
    
    Flujo:
    1. Valida los bits de entrada
    2. Mapea bits a símbolos QPSK
    3. Construye matriz de IFFT (con o sin pilotos)
    4. Modula con IFFT
    5. Añade prefijo cíclico
    6. Serializa
    7. Antepone el preámbulo
    
    Args:
        data_bits: Array 1D de bits (0s y 1s) para el payload
        use_pilots: Si es True, usa pilotos y K_DATA subportadoras
    
    Returns:
        np.ndarray: Señal completa [PREÁMBULO + PAYLOAD] lista para transmitir
    """
    
    # --- 1. Validación y cálculo de parámetros ---
    if use_pilots:
        num_data_bits_per_sym = p.K_DATA * p.mu
    else:
        num_data_bits_per_sym = p.K_TOTAL * p.mu
    
    if len(data_bits) % num_data_bits_per_sym != 0:
        raise ValueError(
            f"El número de bits ({len(data_bits)}) no es múltiplo de "
            f"los bits por símbolo ({num_data_bits_per_sym}). "
            f"use_pilots={use_pilots}"
        )
    
    num_ofdm_symbols = len(data_bits) // num_data_bits_per_sym
    
    # --- 2. Cadena de procesamiento ---
    
    # Bloque 2: Mapeo bits → símbolos
    data_symbols_flat = map_bits_to_symbols(data_bits)
    
    # Bloque 3: Construcción de matriz de IFFT
    if use_pilots:
        X_matrix = build_ifft_input_matrix_with_pilots(data_symbols_flat, num_ofdm_symbols)
    else:
        X_matrix = build_ifft_input_matrix(data_symbols_flat, num_ofdm_symbols)
    
    # Bloque 4: Modulación IFFT
    x_time = modulate_with_ifft(X_matrix)
    
    # Bloque 5: Añadir CP
    x_time_with_cp = add_cyclic_prefix(x_time)
    
    # Bloque 6: Paralelo a Serie
    data_payload_serial = parallel_to_serial(x_time_with_cp)
    
    # --- 3. Ensamblaje del paquete ---
    full_packet_signal = build_full_frame(data_payload_serial)
    
    return full_packet_signal

def create_packets_from_bits(all_data_bits, symbols_per_packet=100, use_pilots=False):
    """
    Divide bits en múltiples paquetes OFDM independientes.
    
    Args:
        all_data_bits: Todos los bits a transmitir
        symbols_per_packet: Símbolos OFDM por paquete
        use_pilots: Si se usan pilotos
    
    Returns:
        list: Lista de paquetes (arrays complejos), cada uno con su preámbulo
    """
    
    # Calcular bits por símbolo según configuración
    if use_pilots:
        bits_per_symbol = p.K_DATA * p.mu
    else:
        bits_per_symbol = p.K_TOTAL * p.mu
    
    # Calcular bits por paquete
    bits_per_packet = symbols_per_packet * bits_per_symbol
    
    # Aplicar padding si es necesario
    total_bits = len(all_data_bits)
    if total_bits % bits_per_packet != 0:
        padding_needed = bits_per_packet - (total_bits % bits_per_packet)
        all_data_bits = np.concatenate([all_data_bits, np.zeros(padding_needed, dtype=int)])
        print(f"⚠ Se añadieron {padding_needed} bits de padding")
    
    # Dividir en chunks
    num_packets = len(all_data_bits) // bits_per_packet
    packets = []
    
    print(f"\n{'='*60}")
    print(f"CREANDO {num_packets} PAQUETES")
    print(f"{'='*60}")
    print(f"Bits totales: {len(all_data_bits)}")
    print(f"Símbolos por paquete: {symbols_per_packet}")
    print(f"Bits por símbolo: {bits_per_symbol}")
    print(f"Bits por paquete: {bits_per_packet}")
    print(f"Usando pilotos: {'Sí' if use_pilots else 'No'}")
    print(f"{'='*60}\n")
    
    for i in range(num_packets):
        # Extraer bits del paquete actual
        start_idx = i * bits_per_packet
        end_idx = start_idx + bits_per_packet
        packet_bits = all_data_bits[start_idx:end_idx]
        
        # Crear el paquete OFDM
        packet_signal = create_packet_from_bits(packet_bits, use_pilots)
        packets.append(packet_signal)
        
        # Información del paquete
        preamble_len = p.N + p.L
        payload_len = symbols_per_packet * (p.N + p.L)
        total_len = preamble_len + payload_len
        
        if i == 0:  # Solo mostrar detalles del primer paquete
            print(f"Estructura del paquete:")
            print(f"  - Preámbulo: {preamble_len} muestras")
            print(f"  - Payload: {payload_len} muestras ({symbols_per_packet} símbolos)")
            print(f"  - Total: {total_len} muestras")
            print(f"  - Real: {len(packet_signal)} muestras\n")
        
        if (i + 1) % 5 == 0 or i == 0 or i == num_packets - 1:
            print(f"✓ Paquete {i+1}/{num_packets} creado")
    
    print(f"\n{'='*60}")
    print(f"✓ {num_packets} paquetes creados exitosamente")
    print(f"{'='*60}\n")
    
    return packets


print("Módulo 'transmisor.py' cargado y listo.")