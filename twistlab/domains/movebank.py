
from __future__ import annotations
import numpy as np
import pandas as pd

def latlon_to_xy(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Simple equirectangular projection for small regions (for demo purposes).
    For production, use pyproj; here we avoid extra deps.
    """
    # constants
    R = 6371000.0
    lat = np.radians(lat); lon = np.radians(lon)
    lat0 = np.nanmean(lat); lon0 = np.nanmean(lon)
    x = R * (lon - lon0) * np.cos(lat0)
    y = R * (lat - lat0)
    return x, y
