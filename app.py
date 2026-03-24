import numpy as np
import pickle
import os
import json
import logging
import tempfile
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf

# ── Logging ───────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# ── Config ────────────────────────────────────────────────────
LOOKBACK_STEPS    = 60
FORECAST_STEPS    = 60
FIREBASE_PATH     = 'devices/esp32_01/readings'
FEATURE_COLUMNS   = ['Aggregate', 'voltage', 'current', 'hour', 'day_of_week']
ANOMALY_THRESHOLD = 2.5
INR_PER_KWH       = 7.0   # ₹ per kWh

# ── Load model + scalers ──────────────────────────────────────
log.info("Loading model and scalers...")
try:
    model          = tf.keras.models.load_model('energy_model.keras')
    feature_scaler = pickle.load(open('feature_scaler(2).pkl', 'rb'))
    target_scaler  = pickle.load(open('target_scaler(2).pkl', 'rb'))
    log.info("✅ Model loaded")
except Exception as e:
    log.error(f"❌ Failed to load model: {e}")
    model = None

# ── Firebase ──────────────────────────────────────────────────
firebase_app = None
try:
    import firebase_admin
    from firebase_admin import credentials, db

    creds_json = os.environ.get('FIREBASE_CREDENTIALS')
    db_url     = os.environ.get('FIREBASE_DB_URL')

    if creds_json and db_url:
        creds_dict = json.loads(creds_json)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            json.dump(creds_dict, tmp)
            tmp_path = tmp.name
        cred = credentials.Certificate(tmp_path)
        firebase_admin.initialize_app(cred, {'databaseURL': db_url})
        firebase_app = True
        log.info("✅ Firebase connected")
    else:
        log.warning("⚠️ Firebase secrets not set")
except Exception as e:
    log.warning(f"⚠️ Firebase init failed: {e}")


# ══════════════════════════════════════════════════════════════
#  CORE FETCH FUNCTION
#  KEY DESIGN DECISIONS:
#  1. Firebase push keys are always chronological — trust that order
#  2. NEVER sort by timestamp — ESP32 sends boot-relative seconds
#  3. ONLY include records that have a real 'power' field
#  4. iloc[-1] = most recent reading (last Firebase key)
# ══════════════════════════════════════════════════════════════

def fetch_firebase_readings(limit=200):
    """
    Fetch latest N readings from Firebase.
    Returns list ordered by insertion time (Firebase key order).
    Skips any record missing the 'power' field (old/incomplete records).
    """
    from firebase_admin import db
    ref      = db.reference(FIREBASE_PATH)
    snapshot = ref.order_by_key().limit_to_last(limit).get()
    if not snapshot:
        return []

    readings = []
    for key, val in sorted(snapshot.items()):  # sorted by Firebase push key = chronological
        if not isinstance(val, dict):
            continue
        # ✅ skip old records that have no real power reading
        if 'power' not in val:
            continue
        power   = float(val.get('power',   0))
        voltage = float(val.get('voltage', 230))
        current = float(val.get('current', 0))
        ts      = val.get('timestamp',     0)
        readings.append({
            'timestamp': ts,
            'Aggregate': power,
            'voltage'  : voltage,
            'current'  : current,
        })
    return readings


def build_features(readings):
    """
    Build feature columns needed by the model.
    Does NOT sort by timestamp — preserves Firebase insertion order.
    Uses reading index to derive hour/day_of_week as approximation
    when real timestamps are unavailable.
    """
    df = pd.DataFrame(readings)

    # try to extract hour/dow from timestamp if it looks like real unix time
    # real unix time for 2020+ = > 1577836800
    def safe_hour(ts):
        try:
            if isinstance(ts, (int, float)) and ts > 1577836800:
                return pd.to_datetime(ts, unit='s').hour
        except Exception:
            pass
        return pd.Timestamp.now().hour   # fallback to current hour

    def safe_dow(ts):
        try:
            if isinstance(ts, (int, float)) and ts > 1577836800:
                return pd.to_datetime(ts, unit='s').dayofweek
        except Exception:
            pass
        return pd.Timestamp.now().dayofweek

    df['hour']        = df['timestamp'].apply(safe_hour)
    df['day_of_week'] = df['timestamp'].apply(safe_dow)
    return df


def run_prediction(df, rolls=4):
    """
    Rolling multi-step forecast.
    rolls=4 → 4 × 8 min = 32 min total forecast.
    """
    if model is None:
        raise RuntimeError("Model not loaded")
    if len(df) < LOOKBACK_STEPS:
        raise ValueError(f"Need {LOOKBACK_STEPS} readings, got {len(df)}")

    window        = df[FEATURE_COLUMNS].values[-LOOKBACK_STEPS:].astype(np.float32)
    window_scaled = feature_scaler.transform(window).copy()
    hour_now      = int(df['hour'].iloc[-1])
    dow_now       = int(df['day_of_week'].iloc[-1])
    all_preds     = []

    for roll in range(rolls):
        X        = window_scaled.reshape(1, LOOKBACK_STEPS, len(FEATURE_COLUMNS))
        y_scaled = model.predict(X, verbose=0)
        y_watts  = target_scaler.inverse_transform(y_scaled)[0]
        all_preds.extend(y_watts.tolist())

        last_row = window_scaled[-1].copy()
        new_rows = []
        for step in range(FORECAST_STEPS):
            row        = last_row.copy()
            row[0]     = y_scaled[0][step]
            total_secs = (roll * FORECAST_STEPS + step) * 8
            row[3]     = ((hour_now * 3600 + total_secs) % 86400) / 3600 / 23.0
            row[4]     = ((dow_now  + (hour_now * 3600 + total_secs) // 86400) % 7) / 6.0
            new_rows.append(row)

        window_scaled = np.vstack([window_scaled[FORECAST_STEPS:],
                                   np.array(new_rows, dtype=np.float32)])

    total_steps   = FORECAST_STEPS * rolls
    total_minutes = round(total_steps * 8 / 60, 1)
    y_all         = np.array(all_preds)

    return {
        'status'           : 'success',
        'forecast_steps'   : total_steps,
        'forecast_minutes' : total_minutes,
        'rolls'            : rolls,
        'predictions_watts': [round(float(v), 2) for v in y_all],
        'summary': {
            'min_watts': round(float(y_all.min()),  2),
            'max_watts': round(float(y_all.max()),  2),
            'avg_watts': round(float(y_all.mean()), 2),
            'total_kwh': round(float(y_all.mean() * (total_steps * 8 / 3600) / 1000), 4)
        }
    }


# ── Routes ────────────────────────────────────────────────────

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'status'            : '✅ Energy Predictor API is live',
        'model_loaded'      : model is not None,
        'firebase_connected': firebase_app is not None,
        'lookback_steps'    : LOOKBACK_STEPS,
        'forecast_steps'    : FORECAST_STEPS,
        'features'          : FEATURE_COLUMNS,
        'endpoints'         : [
            'GET  /',
            'POST /predict',
            'GET  /fetch-and-predict',
            'GET  /history?limit=50',
            'GET  /stats',
            'GET  /anomaly?limit=100',
            'GET  /ui',
        ]
    })


@app.route('/ui')
def ui():
    from flask import send_file
    return send_file('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data     = request.get_json(force=True)
        readings = data.get('readings', [])
        if not readings:
            return jsonify({'error': 'No readings provided'}), 400
        df     = build_features(readings)
        result = run_prediction(df)
        return jsonify(result)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        log.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/fetch-and-predict', methods=['GET'])
def fetch_and_predict():
    if not firebase_app:
        return jsonify({'error': 'Firebase not connected'}), 503
    try:
        readings = fetch_firebase_readings(limit=LOOKBACK_STEPS + 50)
        if not readings:
            return jsonify({'error': 'No data found in Firebase'}), 404
        if len(readings) < LOOKBACK_STEPS:
            return jsonify({'error': f'Need {LOOKBACK_STEPS} readings, got {len(readings)}'}), 400
        df     = build_features(readings)
        result = run_prediction(df)
        result['source']           = 'firebase_live'
        result['readings_fetched'] = len(readings)
        return jsonify(result)
    except Exception as e:
        log.error(f"fetch-and-predict error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/history', methods=['GET'])
def history():
    if not firebase_app:
        return jsonify({'error': 'Firebase not connected'}), 503
    try:
        limit    = min(int(request.args.get('limit', 50)), 500)
        readings = fetch_firebase_readings(limit=limit)
        if not readings:
            return jsonify({'error': 'No data found'}), 404

        records = []
        for i, r in enumerate(readings):
            records.append({
                'index'      : i + 1,
                'timestamp'  : str(r['timestamp']),
                'power_watts': round(float(r['Aggregate']), 2),
                'voltage'    : round(float(r['voltage']),   2),
                'current'    : round(float(r['current']),   4),
                'cost_per_hour': round(float(r['Aggregate']) * INR_PER_KWH / 1000, 4),
            })

        return jsonify({
            'status'  : 'success',
            'count'   : len(records),
            'readings': records,
            'latest'  : records[-1] if records else None,
        })
    except Exception as e:
        log.error(f"history error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/stats', methods=['GET'])
def stats():
    if not firebase_app:
        return jsonify({'error': 'Firebase not connected'}), 503
    try:
        # fetch enough records to cover 1 hour (450 × 8s = 3600s)
        readings = fetch_firebase_readings(limit=500)
        if not readings:
            return jsonify({'error': 'No data found'}), 404

        # ✅ preserve Firebase insertion order — do NOT sort by timestamp
        df         = pd.DataFrame(readings)
        df['power'] = df['Aggregate'].astype(float)

        # use record count for time windows (timestamps are unreliable)
        READINGS_1H  = 450    # 450 × 8s = 1 hour
        READINGS_24H = 10800  # 10800 × 8s = 24 hours

        last_1h  = df.tail(min(READINGS_1H,  len(df)))
        last_24h = df.tail(min(READINGS_24H, len(df)))

        def kwh(subset):
            return round(float(subset['power'].mean()) * len(subset) * 8 / 3600 / 1000, 4)

        # ✅ iloc[-1] = most recent Firebase record
        latest = df.iloc[-1]

        return jsonify({
            'status'            : 'success',
            'current_watts'     : round(float(latest['power']),   2),
            'current_voltage'   : round(float(latest['voltage']), 2),
            'current_current_a' : round(float(latest['current']), 4),
            'last_1h': {
                'avg_watts' : round(float(last_1h['power'].mean()), 2),
                'max_watts' : round(float(last_1h['power'].max()),  2),
                'kwh'       : kwh(last_1h),
                'readings'  : len(last_1h),
            },
            'last_24h': {
                'avg_watts' : round(float(last_24h['power'].mean()), 2),
                'max_watts' : round(float(last_24h['power'].max()),  2),
                'kwh'       : kwh(last_24h),
                'readings'  : len(last_24h),
            },
            'all_time': {
                'avg_watts'     : round(float(df['power'].mean()), 2),
                'max_watts'     : round(float(df['power'].max()),  2),
                'min_watts'     : round(float(df['power'].min()),  2),
                'total_readings': len(df),
            }
        })
    except Exception as e:
        log.error(f"stats error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/anomaly', methods=['GET'])
def anomaly():
    if not firebase_app:
        return jsonify({'error': 'Firebase not connected'}), 503
    try:
        limit    = min(int(request.args.get('limit', 100)), 500)
        readings = fetch_firebase_readings(limit=limit)
        if not readings:
            return jsonify({'error': 'No data found'}), 404

        # preserve insertion order — no timestamp sort
        df         = pd.DataFrame(readings)
        df['power'] = df['Aggregate'].astype(float)
        df['idx']   = range(len(df))

        mean = df['power'].mean()
        std  = df['power'].std()

        if std == 0:
            return jsonify({
                'status': 'ok', 'anomalies': [],
                'baseline_avg_w': round(mean, 2), 'baseline_std_w': 0,
                'threshold_zscore': ANOMALY_THRESHOLD,
                'anomaly_count': 0,
                'message': 'Insufficient variance to detect anomalies'
            })

        df['z_score'] = (df['power'] - mean) / std
        spikes        = df[df['z_score'].abs() > ANOMALY_THRESHOLD]

        anomalies = []
        for _, row in spikes.iterrows():
            anomalies.append({
                'timestamp'  : f"Reading #{int(row['idx']) + 1}",
                'power_watts': round(float(row['power']),   2),
                'z_score'    : round(float(row['z_score']), 2),
                'type'       : 'high_spike' if row['z_score'] > 0 else 'low_drop',
            })

        return jsonify({
            'status'          : 'anomaly_detected' if anomalies else 'ok',
            'anomaly_count'   : len(anomalies),
            'threshold_zscore': ANOMALY_THRESHOLD,
            'baseline_avg_w'  : round(mean, 2),
            'baseline_std_w'  : round(std,  2),
            'anomalies'       : anomalies,
            'message'         : f"{len(anomalies)} spike(s) detected in last {limit} readings"
                                 if anomalies else "All readings normal",
        })
    except Exception as e:
        log.error(f"anomaly error: {e}")
        return jsonify({'error': str(e)}), 500


# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))
    app.run(host='0.0.0.0', port=port, debug=False)