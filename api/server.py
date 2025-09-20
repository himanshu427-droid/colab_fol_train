from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
import torch, joblib, numpy as np, os, time
from models.transformer_model import TransformerAutoencoder
from utils.geo_utils import load_geofences, check_geofence, save_geofences
from utils.feature_utils import compute_features
from shapely.geometry import Polygon, mapping
from datetime import datetime

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins='*')

# Transformer input config
SEQ_LEN = 20
N_FEATURES = 6  # 6 features: dt, dist, speed, bearing, d_bearing, accel

# Load Transformer model
model = TransformerAutoencoder(SEQ_LEN, N_FEATURES)
if os.path.exists('models/artifacts/transformer_ae_v2.pt'):
    model.load_state_dict(torch.load('models/artifacts/transformer_ae_v2.pt', map_location='cpu'))
model.eval()

# Load scaler if exists
scaler = None
if os.path.exists('models/artifacts/scaler_v2.pkl'):
    scaler = joblib.load('models/artifacts/scaler_v2.pkl')

# Load geofences
geofences = load_geofences()

@app.route('/health')
def health():
    return {'status': 'ok'}


@app.route('/predict/anomaly', methods=['POST'])
def predict_anomaly():
    data = request.get_json()

    latitudes = data.get('latitudes')
    longitudes = data.get('longitudes')
    # timestamps = [
    #     datetime.fromisoformat(p["ts"].replace("Z", "+00:00")) for p in seq
    # ]
    timestamps = data.get('timestamps')

    # Compute sequence features
    sequence = compute_features(latitudes, longitudes, timestamps)
    if not sequence:
        return jsonify({'success': False, 'message': 'No valid sequence generated'}), 400

    # Convert to numpy
    raw_features = np.array(
        [[f['dt'], f['dist'], f['speed'], f['bearing'], f['d_bearing'], f['accel']]
         for f in sequence],
        dtype=np.float32
    )

    # Pad sequence to SEQ_LEN if too short
    if raw_features.shape[0] < SEQ_LEN:
        pad_len = SEQ_LEN - raw_features.shape[0]
        padding = np.zeros((pad_len, N_FEATURES), dtype=np.float32)
        raw_features = np.vstack([padding, raw_features])  # prepend zeros

    # Trim sequence if longer than SEQ_LEN
    elif raw_features.shape[0] > SEQ_LEN:
        raw_features = raw_features[-SEQ_LEN:, :]

    # Scale features if scaler exists
    if scaler:
        scaled = scaler.transform(raw_features)
    else:
        scaled = raw_features

    # Feed into transformer
    tensor_input = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)  # shape: [1, SEQ_LEN, N_FEATURES]
    with torch.no_grad():
        reconstructed = model(tensor_input)
        mse = torch.mean((tensor_input - reconstructed) ** 2).item()
    
    print(mse)

    return jsonify({
        'success': True,
        'sequence_length': len(sequence),
        'anomaly_score': mse,
        'type': 'emergency',
        'isAnomaly': True if mse>70 else False,
        'score': 1 if mse>70 else 0
    })


@app.route('/ingest/ping', methods=['POST'])
def ingest_ping():
    """ Receives tourist's current location from Node.js frontend or mobile app. """
    data = request.get_json()
    tourist_id = data.get('touristId')
    lat = float(data.get('lat'))
    lon = float(data.get('lon'))
    ts = data.get('ts', time.time())


    g = check_geofence(lat, lon, geofences)
    actions = []


    if g:
        actions.append({
            'geofence': g['name'],
            'restricted': g.get('restricted', False)
        })

        # Alert if restricted
        if g.get('restricted', False):
            alert = {
                'type': 'geofence_restricted',
                'tourist_id': tourist_id,
                'geofence': g,
                'ts': ts
            }
            # socketio.emit('geofence_alert', alert)

    # # Broadcast location to dashboard or families
    # socketio.emit('location_update', {
    #     'tourist_id': tourist_id,
    #     'lat': lat,
    #     'lon': lon,
    #     'ts': ts
    # })

    return jsonify({'status': 'ok', 'actions': actions})

@app.route('/add_geofence', methods=['POST'])
def add_geofence():
    data = request.get_json()
    geofences = load_geofences()

    if isinstance(geofences, dict):
        geofences = [geofences]

    # Clean up invalid 'polygon' fields
    cleaned = []
    for g in geofences:
        if isinstance(g, dict) and "polygon" in g:
            g.pop("polygon")  # remove the polygon key
        cleaned.append(g)


    geofences.append(data)
    save_geofences(geofences)
    return jsonify({'status': 'ok', 'new_geofence': data, 'total_geofences': len(geofences)})

    
@app.route('/panic', methods=['POST'])
def panic():
    data = request.get_json(force=True)
    alert = {'type': 'panic', 'data': data, 'ts': time.time()}
    socketio.emit('panic', alert)
    return jsonify({'status': 'ok', 'alert': alert})


@socketio.on('subscribe')
def subscribe(data):
    tourist_id = data.get('tourist_id')
    emit('subscribed', {'tourist_id': tourist_id})


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=8080, debug=True)

