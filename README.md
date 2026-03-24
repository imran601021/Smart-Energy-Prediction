# ⚡ Smart Energy Conservation System

> Real-time home energy monitoring with LSTM-based power consumption forecasting, anomaly detection, and a live web dashboard — powered by ESP32, Firebase, and Hugging Face.

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)
![Flask](https://img.shields.io/badge/Flask-API-lightgrey?style=flat-square&logo=flask)
![Firebase](https://img.shields.io/badge/Firebase-RTDB-yellow?style=flat-square&logo=firebase)
![ESP32](https://img.shields.io/badge/ESP32-MicroPython-red?style=flat-square)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Spaces-ff9d00?style=flat-square)

---

## 📌 Overview

This is a final-year IoT + Machine Learning project that monitors household electricity consumption in real time, forecasts future usage using an LSTM neural network, and detects power anomalies — all visualized through a live web dashboard.

The system runs end-to-end:

```
ESP32 Sensors → Firebase RTDB → Flask API (HF Spaces) → Live Web Dashboard
```

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     HARDWARE LAYER                      │
│  ACS712-20A          ZMPT101B           ESP32 DevKit    │
│  Current Sensor  +   Voltage Sensor  →  MicroPython     │
└────────────────────────┬────────────────────────────────┘
                         │ HTTP POST (JSON) every 10s
                         ▼
┌─────────────────────────────────────────────────────────┐
│                    FIREBASE RTDB                        │
│  devices/esp32_01/readings/{push_key}                   │
│  { power, voltage, current, timestamp, cost_per_hour }  │
└────────────────────────┬────────────────────────────────┘
                         │ Firebase Admin SDK
                         ▼
┌─────────────────────────────────────────────────────────┐
│              FLASK API — Hugging Face Spaces            │
│  /history      → last N raw readings                    │
│  /stats        → live + 1h + 24h + all-time summary     │
│  /anomaly      → z-score spike detection                │
│  /fetch-and-predict → LSTM rolling forecast             │
│  /ui           → serves web dashboard                   │
└────────────────────────┬────────────────────────────────┘
                         │ REST API (CORS enabled)
                         ▼
┌─────────────────────────────────────────────────────────┐
│               WEB DASHBOARD (index.html)                │
│  Page 1: Login                                          │
│  Page 2: Live Dashboard (auto-refresh every 10s)        │
│  Page 3: 30-min LSTM Power Forecast                     │
└─────────────────────────────────────────────────────────┘
```

---

## ✨ Features

### Hardware
- Real-time AC current measurement using ACS712-20A (100mV/A, 3.3V powered)
- AC voltage measurement using ZMPT101B sensor
- RMS sampling over exactly 2 full 50Hz cycles (800 samples / 40ms)
- NTP time sync for accurate unix timestamps
- Auto-recalibration after 12.5 minutes of zero current
- Watchdog timer for reliable 24/7 operation

### Backend API
| Endpoint | Description |
|---|---|
| `GET /` | Health check — model + Firebase status |
| `GET /history?limit=N` | Last N raw sensor readings |
| `GET /stats` | Live power, 1h avg/kWh, 24h summary, all-time peak |
| `GET /anomaly?limit=N` | Z-score spike detection (threshold: ±2.5σ) |
| `GET /fetch-and-predict` | Live LSTM rolling forecast |
| `GET /ui` | Serves the web dashboard |

### Machine Learning
- LSTM model trained on 4.4M rows of real household power data
- Features: `Aggregate`, `voltage`, `current`, `hour`, `day_of_week`
- Rolling multi-step inference: 4 rolls × 8 min = **32-minute forecast**
- No retraining needed for extended forecasts — model output feeds back as input
- Training uses `tf.keras.utils.timeseries_dataset_from_array` with `sequence_stride=3` to prevent Colab RAM crashes

### Web Dashboard
- **Login page** — credential-based access
- **Live dashboard** — auto-refreshes every 10s with countdown ring
  - Power, Voltage, Current stat cards
  - 1h energy (kWh) and cost estimate in ₹
  - Live power line chart (in-place update, no flicker)
  - Power distribution doughnut (High/Normal/Low)
  - Anomaly table with z-score and spike type
- **Prediction page** — 30-min LSTM forecast
  - 60-step forecast chart with avg reference line
  - Cost estimate in ₹ for forecast period

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Microcontroller | ESP32 DevKit V1 |
| Firmware | MicroPython v3.0 |
| Current Sensor | ACS712-20A |
| Voltage Sensor | ZMPT101B |
| Cloud Database | Firebase Realtime Database |
| ML Training | Google Colab + TensorFlow 2.x |
| ML Model | LSTM (128→64 units, Huber loss) |
| API | Flask + Flask-CORS |
| API Hosting | Hugging Face Spaces (Docker) |
| Frontend | Vanilla HTML/CSS/JS + Chart.js |
| Version Control | Git (GitHub + Hugging Face) |

---

## 📁 Project Structure

```
Smart-Energy-Conservation/
├── app.py                  # Flask API — all endpoints
├── index.html              # 3-page web dashboard
├── main.py                 # ESP32 MicroPython firmware
├── Dockerfile              # HF Spaces container config
├── requirements.txt        # Python dependencies
├── energy_model.keras      # Trained LSTM model
├── feature_scaler.pkl      # MinMaxScaler for features
├── target_scaler.pkl       # MinMaxScaler for target
└── README.md
```

---

## 🚀 Getting Started

### 1. Hardware Setup

```
ESP32 GPIO34  ← ACS712 OUT  (current sensor)
ESP32 GPIO35  ← ZMPT101B OUT (voltage sensor)
ESP32 3.3V    → ACS712 VCC + ZMPT101B VCC
ESP32 GND     → ACS712 GND + ZMPT101B GND
ESP32 GPIO2   → LED (status indicator)
```

> ⚠️ Power both sensors from **3.3V**, not 5V. The ESP32 ADC is 3.3V — 5V will damage it.

### 2. Flash Firmware

1. Install [Thonny IDE](https://thonny.org/)
2. Flash MicroPython to ESP32
3. Edit `main.py` — set your WiFi credentials and Firebase URL
4. Upload `main.py` to ESP32 root
5. Run — check serial monitor for `✓ Time synced` and `✓ Firebase 200`

### 3. Deploy API

```bash
# clone repo
git clone https://huggingface.co/spaces/CallMeRolex/Energy-Prediction
cd Energy-Prediction

# set HF Space secrets
# FIREBASE_CREDENTIALS = { ...service_account.json content... }
# FIREBASE_DB_URL      = https://your-project-default-rtdb.firebaseio.com
```

### 4. Access Dashboard

```
https://callmerolex-energy-prediction.hf.space/ui
```

Login: `admin` / `energy123`

---

## 🧠 Model Details

| Parameter | Value |
|---|---|
| Architecture | LSTM (128) → Dropout(0.2) → LSTM(64) → Dense(64) → Dense(32) → Dense(60) |
| Input shape | (60, 5) — 60 timesteps × 5 features |
| Output shape | (60,) — 60-step forecast |
| Loss function | Huber (robust to spikes) |
| Optimizer | Adam (lr=1e-3) with ReduceLROnPlateau |
| Training data | 4,431,533 rows — UK household power dataset |
| Sampling interval | 8 seconds |
| Forecast method | Rolling multi-step (4 rolls × 8 min = 32 min) |

---

## 📊 API Response Examples

**`GET /stats`**
```json
{
  "status": "success",
  "current_watts": 66.37,
  "current_voltage": 207.67,
  "current_current_a": 0.376,
  "last_1h": { "avg_watts": 410.01, "kwh": 0.4383, "readings": 450 },
  "all_time": { "max_watts": 3374.33, "total_readings": 5000 }
}
```

**`GET /fetch-and-predict`**
```json
{
  "status": "success",
  "forecast_minutes": 32.0,
  "forecast_steps": 240,
  "rolls": 4,
  "predictions_watts": [978.23, 818.9, 948.9, "..."],
  "summary": { "avg_watts": 896.23, "total_kwh": 0.1195 }
}
```

---

## ⚠️ Known Limitations

- ESP32 sends boot-relative timestamps until NTP syncs — dashboard uses record-count windowing as fallback
- Rolling forecast accuracy degrades beyond 2 rolls as prediction error compounds
- ACS712 has ~12mA resolution at 12-bit — not suitable for very low power loads (<5W)
- HF Spaces free tier may cold-start (30s delay on first request)

---

## 🔮 Future Work

- Retrain model with 15 features and LOOKBACK=90 for better accuracy
- Fix NTP sync issue in firmware for accurate time-based filtering
- Add appliance-level disaggregation (NILM)
- Mobile app with push notifications for anomalies
- Solar panel integration for net metering

---

## 👨‍💻 Author

**Imran** — Final Year B.E. Project, 2026

- 🔗 Live Demo: [https://callmerolex-energy-prediction.hf.space/ui](https://callmerolex-energy-prediction.hf.space/ui)
- 🤗 HF Space: [https://huggingface.co/spaces/CallMeRolex/Energy-Prediction](https://huggingface.co/spaces/CallMeRolex/Energy-Prediction)

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.
