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





