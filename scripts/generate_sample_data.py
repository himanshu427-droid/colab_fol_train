import os
from datetime import datetime
from utils.feature_utils import compute_features

def read_plt_file(filepath):
    latitudes, longitudes, timestamps = [], [], []
    with open(filepath, 'r') as f:
        lines = f.readlines()[6:]  # skip 6 header lines in Geolife
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) < 7:
                continue
            lat, lon = float(parts[0]), float(parts[1])
            date, time = parts[5], parts[6]
            ts = datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M:%S")
            latitudes.append(lat)
            longitudes.append(lon)
            timestamps.append(ts)
    return latitudes, longitudes, timestamps


def extract_features_from_file(filepath):
    lats, lons, ts = read_plt_file(filepath)
    if len(lats) < 2:
        return []
    features = compute_features(lats, lons, ts)
    return features