# ESP32 TinyML: End-to-End Edge AI Pipeline

![ESP-IDF](https://img.shields.io/badge/ESP--IDF-v5.0%2B-blue?logo=espressif)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Language](https://img.shields.io/badge/C%2B%2B-17-green?logo=cplusplus)
![Python](https://img.shields.io/badge/Python-3.9%2B-yellow?logo=python)

An end-to-end TinyML (Edge AI) implementation demonstrating how to train a neural network in TensorFlow/Keras, apply full **INT8 Post-Training Quantization (PTQ)**, export the quantized model as a C byte array, and deploy it onto an **ESP32 microcontroller** using **TensorFlow Lite Micro (TFLM)** and **ESP-IDF**.

---

## 📌 Project Overview

This repository demonstrates the complete lifecycle of an embedded Machine Learning application:

```
┌────────────────────────────────┐     ┌───────────────────────────────┐     ┌───────────────────────────────┐
│     1. Train Keras Model       │ ──> │ 2. Full INT8 PTQ Quantization │ ──> │   3. Export to model_data.h   │
│       y = 3x + 1 (Dense)       │     │     TFLite Converter & Rep.   │     │    C Byte Array (`g_model`)   │
└────────────────────────────────┘     └───────────────────────────────┘     └───────────────────────────────┘
                                                                                             │
                                                                                             ▼
┌────────────────────────────────┐     ┌───────────────────────────────┐     ┌───────────────────────────────┐
│    6. Real-Time Inference      │ <── │   5. Allocate Tensors & Arena │ <── │   4. ESP32 ESP-IDF Firmware   │
│ Latency & Dequantized Output   │     │      4 KB Static RAM Arena    │     │  TensorFlow Lite Micro Engine │
└────────────────────────────────┘     └───────────────────────────────┘     └───────────────────────────────┘
```

---

## 📁 Repository Structure

```
tinyml_workspace/
├── README.md               # Complete project documentation & guide
├── model_data.h            # Exported C byte array header (root copy)
├── src/                    # Machine Learning Training & Export Pipeline
│   ├── main.py             # Keras training, INT8 quantization, & C header generator script  
│   ├── model.tflite        # Generated INT8 quantized TensorFlow Lite FlatBuffer
│   └── model_data.h        # Exported C byte array header (`g_model`)
└── esp32_tinyml/           # ESP32 Embedded Firmware Project (ESP-IDF)
    ├── CMakeLists.txt      # Root CMake configuration file
    ├── sdkconfig           # ESP-IDF target SDK configuration
    ├── main/               # Application source directory
    │   ├── CMakeLists.txt  # Main component build script (C++17, dependencies)
    │   ├── idf_component.yml # Component manager specification for TFLM
    │   ├── main.cc         # C++ application entry point (tflite interpreter & inference loop)
    │   └── model_data.h    # Embedded model data header included by main.cc
    └── managed_components/ # Managed ESP-IDF components (`espressif__esp-tflite-micro`)
```

---

## 🧠 Machine Learning Model & Quantization

### Model Architecture (`src/main.py`)
The model fits a linear equation $y = 3x + 1$ with artificial Gaussian noise:
- **Input Layer:** 1D Scalar feature ($x \in [-10, 10]$)
- **Hidden Layer:** `Dense(16)` with `ReLU` activation
- **Output Layer:** `Dense(1)` (Linear regression output)

### Full INT8 Post-Training Quantization (PTQ)
To optimize memory footprint and execution speed on microcontrollers without hardware FPUs or with constrained RAM:
- **Representative Dataset Generator (`rep_dataset`):** Feeds sample inputs during conversion so TensorFlow can compute scaling factors ($S$) and zero-points ($Z$).
- **Fixed-Point Specification:** Restricts operators to `TFLITE_BUILTINS_INT8` and sets both input and output types to `tf.int8`.
- **FlatBuffer Size:** ~2.2 KB.

---

## ⚡ ESP32 Embedded Firmware (`esp32_tinyml/`)

The firmware is built using **ESP-IDF** and the official **`espressif__esp-tflite-micro`** component:

1. **Model Loader:** Loads `g_model` from Flash memory and verifies FlatBuffer schema compatibility (`TFLITE_SCHEMA_VERSION`).
2. **Op Resolver:** Registers operators via `tflite::MicroMutableOpResolver<1>` (`AddFullyConnected`).
3. **Static Tensor Arena:** Allocates a 4 KB static memory pool (`tensor_arena`) to avoid heap fragmentation during inference.
4. **Quantization & De-Quantization at Runtime:**
   - **Quantize Input:** Converts float $x$ to INT8 format:
     $$q_{in} = \text{round}\left(\frac{x_{\text{raw}}}{\text{scale}_{in}}\right) + \text{zero\_point}_{in}$$
   - **Inference:** Calls `interpreter.Invoke()`.
   - **Dequantize Output:** Converts INT8 output back to float:
     $$y_{\text{pred}} = (q_{out} - \text{zero\_point}_{out}) \times \text{scale}_{out}$$
5. **High-Precision Latency Tracking:** Uses `esp_timer_get_time()` to measure exact microsecond execution time.

---

## 🛠️ Requirements & Setup

### 1. Python Environment
- Python 3.9+
- TensorFlow 2.x
- NumPy

Install required Python packages:
```bash
pip install tensorflow numpy
```

### 2. ESP-IDF Development Environment
- **ESP-IDF:** v5.0 or later (v6.0 supported)
- **Target Microcontroller:** ESP32 (Xtensa LX6)

Ensure ESP-IDF environment variables are exported in your terminal session:
```bash
. $HOME/esp/esp-idf/export.sh
```

---

## 🚀 Step-by-Step Guide

### Step 1: Train & Export Quantized Model

Run the Python training script to generate the quantized model and C header:

```bash
cd /home/bava/tinyml_workspace
python src/main.py
```

**Expected Output:**
```
Starting the training of the model
Epoch 200/200 - loss: 0.0102
Initialising the converter
Success: Saved model.tflite
Success: Exported 2232 bytes to model_data.h
```

---

### Step 2: Copy Header File to ESP32 Component

Copy the generated `model_data.h` into the ESP32 project's `main` directory:

```bash
cp src/model_data.h esp32_tinyml/main/model_data.h
```

---

### Step 3: Configure & Build ESP32 Firmware

Navigate to the ESP32 project directory and build using `idf.py`:

```bash
cd /home/bava/tinyml_workspace/esp32_tinyml

# Export ESP-IDF environment (if not done already)
. $HOME/esp/esp-idf/export.sh

# Build the project
idf.py build
```

---

### Step 4: Flash Firmware & Monitor Logs

Connect your ESP32 board via USB and run:

```bash
# Replace /dev/ttyUSB0 with your actual serial port (e.g. COM3 on Windows or /dev/ttyACM0)
idf.py -p /dev/ttyUSB0 flash monitor
```

---

## 📊 Sample Output Log

When the ESP32 runs the model, the serial monitor will display predictions along with measured microsecond latencies:

```text
I (312) bavaai: Starting TinyML Model...
I (332) bavaai: Interpreter ready. Arena used: 1488 bytes
I (332) bavaai: Model expects INT8 inputs. Training equation: y = 3x + 1
I (332) bavaai: Input X:   1.0 | Pred Y:   4.02 (True:   4.00) | Latency: 42 us
I (1332) bavaai: Input X:   2.0 | Pred Y:   7.01 (True:   7.00) | Latency: 41 us
I (2332) bavaai: Input X:   3.0 | Pred Y:  10.05 (True:  10.00) | Latency: 41 us
I (3332) bavaai: Input X:   5.0 | Pred Y:  16.03 (True:  16.00) | Latency: 41 us
I (4332) bavaai: Input X:  10.0 | Pred Y:  31.02 (True:  31.00) | Latency: 42 us
I (5332) bavaai: Inference completed.
```

---

## 🔧 Build Configuration Details

- **C++ Standard:** Must be set to `-std=gnu++17` in `esp32_tinyml/main/CMakeLists.txt` because TensorFlow Lite Micro relies on C++17 features (`if constexpr`, `std::is_integral_v`).
- **Component Dependencies:** `main/CMakeLists.txt` requires `"esp-tflite-micro"` and `"esp_timer"`.
- **Status Enum:** TfLite API uses `kTfLiteOk` for successful execution status.

---

## 📜 License

This project is open-source under the MIT License.
