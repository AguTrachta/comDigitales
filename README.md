# Proyecto OFDM – Trabajo Final

Este repositorio contiene una **cadena OFDM modular** implementada en Python/Jupyter.  La estructura está pensada para que cualquier miembro del equipo pueda:

* localizar rápidamente cada bloque (generación de bits, IFFT, AWGN …),
* ejecutar sólo las secciones que necesite para probar cambios,
* versionar código core separado de los cuadernos de experimentación.

---

## Árbol de carpetas

```text
ofdm-trabajo-final/
│
├── .venv/                # entorno virtual (IGNORADO por git)
├── pyproject.toml        # dependencias y metadatos del paquete
├── README.md             # (este archivo)
│
├── src/                  # código reusable, instalable como paquete
│   └── ofdm_tf/
│       ├── params.py     # PARÁMETROS globales (Bloque 0)
│       ├── mapping.py    # QPSK, 16‑QAM, Gray, etc.
│       ├── fft.py        # IFFT/FFT wrappers y normalizaciones
│       ├── channel.py    # canal AWGN / multipath
│       ├── utils.py      # helpers (Eb/N0, BER, dB<->lin…)
│       └── tests/        # unit tests (pytest)
│
├── notebooks/            # demostraciones paso a paso
│   ├── 00_parametros.ipynb
│   ├── 01_generador_bits.ipynb
│   ├── …
│   └── 12_ber_awgn.ipynb
│
├── data/
│   ├── raw/              # capturas IQ u originales grandes
│   └── processed/        # .npy / .csv intermedios
│
├── figures/              # constelaciones y curvas BER exportadas
│
└── docs/                 # PDFs de referencia / informe escrito
    ├── Dialnet-ModulacionMultiportadoraOFDM-4797263.pdf
    └── plan_de_ataque.pdf
```

> **Instalación rápida**
>
> ```bash
> # activa tu .venv primero
> python -m pip install -e .            # modo editable → importa ofdm_tf
> pip install -r requirements.txt       # si no usas pyproject
> ```

---

## Bloques de desarrollo

| #  | Notebook                  | Módulo clave            | Objetivo                    | Validación mínima        |
| -- | ------------------------- | ----------------------- | --------------------------- | ------------------------ |
| 0  | `00_parametros.ipynb`     | `src/ofdm_tf/params.py` | Definir N, M, L\_cp, etc.   | `assert Δf·T_u = 1`      |
| 1  | `01_generador_bits.ipynb` | `utils.random_bits()`   | Secuencia binaria aleatoria | Histograma 0 ≈ 1         |
| 2  | `02_mapping.ipynb`        | `mapping.gray_map()`    | Mapear bits → símbolos      | Scatter QPSK normalizado |
| 3  | `03_subcarriers.ipynb`    | —                       | Vector X\[k] tamaño N       | Print primeros 8 valores |
| 4  | `04_ifft.ipynb`           | `fft.ifft_block()`      | Señal tiempo x\[n]          | `np.fft.fft(ifft(X))≅X`  |
| 5  | `05_cyclic_prefix.ipynb`  | `utils.add_cp()`        | Insertar prefijo L\_cp      | Long. = N+L\_cp          |
| 6  | `06_serial_tx.ipynb`      | —                       | Trama continua              | Onda sin saltos          |
| 7  | `07_channel_ideal.ipynb`  | `channel.identity()`    | Pasar por canal ideal       | BER = 0                  |
| 8  | `08_remove_cp.ipynb`      | `utils.remove_cp()`     | Cortar CP                   | Señal igual a x\[n]      |
| 9  | `09_fft_rx.ipynb`         | `fft.fft_block()`       | Recuperar Y\[k]             | Y ≅ X                    |
| 10 | `10_demapping.ipynb`      | `mapping.gray_demap()`  | Símbolo → bits              | BER = 0                  |
| 11 | `11_awgn.ipynb`           | `channel.add_awgn()`    | Añadir ruido Eb/N0          | Scatter vs SNR           |
| 12 | `12_ber_curve.ipynb`      | `utils.ber_theory()`    | BER simulado vs teórico     | Curva semilog            |

**Hitos sugeridos**

* **Semana 1** → Bloques 0–5 funcionando (IFFT con CP, sin canal).
* **Semana 2** → Cadena tx/rx ideal completa (BER 0).
* **Semana 3** → AWGN + curva BER comparada con teoría.

---

## Buenas prácticas rápidas

* **Un solo origen de la verdad**: todos los cuadernos importan `ofdm_tf.params`  ⇒ cambia una vez y se propaga.
* **Tests antes de cada commit**: `pytest -q` debe estar limpio.
* **nbstripout**: evita subir salidas enormes de notebooks.
* **figures/**: guarda las imágenes con nombres auto‑explicativos (`ber_qpsk.png`).

¡Listo!  Con esta guía tu compañero puede clonar el repo, instalar dependencias y empezar por el notebook 00 sin preguntarte dónde está cada cosa.
