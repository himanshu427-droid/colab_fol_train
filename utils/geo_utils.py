import json
import os
from shapely.geometry import Point, Polygon, mapping
from utils.feature_utils import haversine

from shapely.geometry import Polygon, mapping

def polygon_to_json(poly: Polygon):
    return mapping(poly)  # GeoJSON dict

def load_geofences(path='data/geofences.json'):
    with open(path, 'r') as f:
        geofences = json.load(f)
    
    
    for g in geofences:
        if g.get('type') == 'polygon' and 'vertices' in g:
            g['polygon'] = Polygon([(v[1], v[0]) for v in g['vertices']])
    return geofences


def check_geofence(lat, lon, geofences):
    point = Point(lon, lat)  
    
    for g in geofences:
        if g['type'] == 'circle':
            dist = haversine(lat, lon, g['center'][0], g['center'][1])
            if dist <= g['radius']:
                return g
        elif g['type'] == 'polygon':
            if g['polygon'].contains(point):
                return g
    return None

def save_geofences(geofences, path='data/geofences.json'):
    """Write geofences list/dict back to JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)  # ensure folder exists
    with open(path, 'w') as f:
        json.dump(geofences, f, indent=2)