# Simulación de un Sistema OFDM Completo

Este repositorio contiene la implementación y simulación de una cadena de transmisión y recepción **OFDM (Orthogonal Frequency Division Multiplexing)**, desarrollada en Python utilizando Jupyter Notebook. El proyecto sigue una estructura modular y está diseñado para ser reproducible y fácil de entender, sirviendo como material de estudio y experimentación.

El código se basa en los principios teóricos descritos en papers académicos sobre OFDM, como el de Göran Lindell, adaptando los conceptos a una implementación práctica y verificable.

---

## 🏗️ Estructura del Proyecto

El repositorio está organizado para separar el código reutilizable de la experimentación, facilitando el mantenimiento y la colaboración.

```
ofdm-trabajo-final/
│
├── .venv/                   # Entorno virtual de Python (ignorado)
├── pyproject.toml           # Dependencias y metadatos del proyecto
├── README.md                # Este archivo
│
├── src/                     # Código fuente del paquete 'ofdm_tf'
│   └── ofdm_tf/
│       ├── __init__.py
│       ├── params.py        # PARÁMETROS globales de la simulación
│       ├── mapping.py       # Mapeo/Demapeo de constelaciones (QPSK)
│       ├── fft.py           # Funciones para IFFT/FFT
│       ├── channel.py       # Modelos de canal (AWGN, etc.)
│       └── utils.py         # Funciones auxiliares (BER, dB, etc.)
│
├── notebooks/               # Cuadernos de experimentación
│   └── ofdm_simulation.ipynb  # NOTEBOOK PRINCIPAL con todos los bloques
│
├── data/                    # Datos (si aplica)
│   ├── raw/
│   └── processed/
│
├── figures/                 # Gráficos y figuras generadas
│
└── docs/                    # Documentación y papers de referencia
```
### 🚀 Instalación Rápida

Para ejecutar la simulación, clona el repositorio, activa tu entorno virtual y ejecuta:

```bash
# Instala el paquete 'ofdm_tf' en modo editable
python -m pip install -e .
```

El modo editable (`-e`) permite que los cambios que hagas en el código de la carpeta `src/` se reflejen inmediatamente en las importaciones del notebook sin necesidad de reinstalar.

---

## 📖 Plan de Desarrollo: Bloques Funcionales

Aunque todo el flujo se implementa en un único notebook (`ofdm_simulation.ipynb`), el desarrollo sigue una secuencia de bloques lógicos bien definidos, cada uno con un objetivo y una forma de validación clara.

| #  | Bloque Funcional          | Módulo Clave            | Objetivo Principal                                    | Validación Mínima                                    |
| -- | ------------------------- | ----------------------- | ----------------------------------------------------- | ---------------------------------------------------- |
| 0  | **Configuración**         | `src/ofdm_tf/params.py` | Definir N, K, L, etc. y sus derivados.                | `assert` de coherencia en los parámetros.            |
| 1  | **Generador de Bits**     | `numpy.random`          | Crear la secuencia de datos binarios de origen.       | Histograma de bits con distribución ~50/50.          |
| 2  | **Mapeo de Símbolos**     | `mapping.py`            | Convertir bits a símbolos de constelación QPSK.       | Diagrama de constelación normalizado ($E_s=1$).      |
| 3  | **Mapeo a Subportadoras** | —                       | Construir el vector de frecuencia `X[k]` (tamaño N).  | Inspección visual del vector con símbolos y ceros.   |
| 4  | **Modulación IFFT**       | `fft.py`                | Transformar del dominio de la frecuencia al tiempo `x[n]`. | `fft(ifft(X)) ≅ X` (propiedad de la transformada). |
| 5  | **Prefijo Cíclico (CP)**  | `utils.py`              | Añadir el prefijo cíclico para mitigar la ISI.        | Longitud del símbolo = `N + L`.                      |
| 6  | **Serialización**         | `numpy.reshape`         | Concatenar símbolos para crear la trama a transmitir. | Verificación de la longitud total de la trama.       |
| 7  | **Paso por Canal**        | `channel.py`            | Simular el efecto del canal (ideal o con ruido).      | **Canal Ideal:** BER = 0.                              |
| 8  | **Eliminación del CP**    | `utils.py`              | Descartar el prefijo cíclico en el receptor.          | Longitud del bloque recibido = `N`.                  |
| 9  | **Demodulación FFT**      | `fft.py`                | Transformar de vuelta al dominio de la frecuencia `Y[k]`. | **Canal Ideal:** `Y[k] ≅ X[k]`.                        |
| 10 | **Demapeo de Símbolos**   | `mapping.py`            | Decidir los bits a partir de los símbolos recibidos.  | **Canal Ideal:** Bits recibidos = Bits transmitidos. |
| 11 | **Simulación con Ruido**  | `channel.py`            | Añadir Ruido Gaussiano Blanco Aditivo (AWGN).         | Diagrama de constelación ruidoso, varianza correcta. |
| 12 | **Curva BER**             | `utils.py`              | Medir el rendimiento (BER) vs. $E_b/N_0$.             | Curva simulada vs. curva teórica de QPSK.            |

### ✅ Hitos Clave

1.  **Transmisor Funcional:** Implementar los bloques 0 a 6. El resultado es una trama OFDM completa lista para ser transmitida.
2.  **Cadena Ideal Tx/Rx:** Completar los bloques 7 a 10 con un canal ideal. El objetivo es lograr una Tasa de Error de Bit (BER) de cero, validando la lógica de la cadena.
3.  **Simulación de Rendimiento:** Implementar los bloques 11 y 12 para analizar el sistema en un canal con ruido (AWGN) y comparar los resultados con la teoría.

---

## 🛠️ Buenas Prácticas

*   **Fuente Única de Verdad:** Todos los parámetros se importan desde `src/ofdm_tf/params.py`. Un cambio en este archivo se propaga a toda la simulación.
*   **Código Reutilizable:** Las funciones complejas o repetitivas se alojan en los módulos de `src/` para mantener el notebook limpio y centrado en la experimentación.
*   **Nombres Descriptivos:** Las figuras y artefactos generados se guardan con nombres que explican su contenido (ej: `ber_qpsk_awgn.png`).
*   **Control de Versiones Limpio:** Se recomienda configurar `nbstripout` para evitar subir los outputs de los notebooks al repositorio Git, manteniendo los commits ligeros.

```
```
