import os
import glob
import pandas as pd
from datetime import datetime

import numpy as np
import math
from datetime import datetime

# Haversine distance between two GPS points (meters)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371e3
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2*math.asin(math.sqrt(a))
    return R * c

# Bearing (direction) from point1 to point2
def bearing(lat1, lon1, lat2, lon2):
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_lon = math.radians(lon2 - lon1)
    x = math.sin(delta_lon) * math.cos(phi2)
    y = math.cos(phi1)*math.sin(phi2) - math.sin(phi1)*math.cos(phi2)*math.cos(delta_lon)
    brng = math.atan2(x, y)
    return (math.degrees(brng) + 360) % 360

# Generate 6-feature sequence from lat/lon/timestamp lists
def compute_features(latitudes, longitudes, timestamps):
    if isinstance(timestamps[0], str):
        timestamps = [datetime.fromisoformat(ts.replace("Z", "+00:00")) for ts in timestamps]

    sequence = []
    for i in range(1,len(latitudes)):
        dt = (timestamps[i] - timestamps[i-1]).total_seconds()
        dist = haversine(latitudes[i-1], longitudes[i-1], latitudes[i], longitudes[i])
        speed = dist / dt if dt != 0 else 0
        brng = bearing(latitudes[i-1], longitudes[i-1], latitudes[i], longitudes[i])
        prev_brng = bearing(latitudes[i-2], longitudes[i-2], latitudes[i-1], longitudes[i-1]) if i > 1 else brng
        d_brng = (brng - prev_brng + 180) % 360 - 180
        prev_speed = sequence[-1]['speed'] if i > 1 else speed
        accel = (speed - prev_speed) / dt if dt != 0 else 0

        sequence.append({
            'dt': dt,
            'dist': dist,
            'speed': speed,
            'bearing': brng,
            'd_bearing': d_brng,
            'accel': accel
        })
    
    return sequence

    # target_len = 20
    # if len(sequence) < target_len:
    #     last = sequence[-1] if sequence else {
    #         'dt': 0, 'dist': 0, 'speed': 0,
    #         'bearing': 0, 'd_bearing': 0, 'accel': 0
    #     }
    #     while len(sequence) < target_len:
    #         sequence.append(last.copy())



def load_plt(path):
    """
    Load a single Geolife trajectory file (.plt)
    """
    df = pd.read_csv(
        path,
        skiprows=6,  # skip header metadata
        names=["lat", "lon", "unused1", "alt", "days", "date", "time"]
    )
    df["ts"] = pd.to_datetime(df["date"] + " " + df["time"])
    return df[["lat", "lon", "ts"]]

def process_trajectory(path):
    df = load_plt(path)
    lats = df["lat"].tolist()
    lons = df["lon"].tolist()
    ts = df["ts"].tolist()
    return compute_features(lats, lons, ts)  # list of dicts

def build_dataset(base_dir="../data/Geolife Trajectories 1.3/Data", out_csv="../data/geolife_features.csv"):
    all_features = []
    user_dirs = [os.path.join(base_dir, d, "Trajectory") for d in os.listdir(base_dir) if d.isdigit()]

    for udir in user_dirs:
        for file in glob.glob(os.path.join(udir, "*.plt")):
            try:
                feats = process_trajectory(file)
                all_features.extend(feats)
            except Exception as e:
                print(f"Skipping {file}, error={e}")

    df = pd.DataFrame(all_features)
    os.makedirs("data", exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"✅ Saved dataset with {len(df)} rows to {out_csv}")

if __name__ == "__main__":
    build_dataset()
