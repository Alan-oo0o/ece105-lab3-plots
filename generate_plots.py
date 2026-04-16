# Intent: Generate synthetic sensor data for lab3 plots
# Sensor A: mean 25 C, std 3 C, 200 readings.
# Sensor B: mean 27 C, std 4.5 C, 200 readings.
# Also generate 200 timestamps uniformly from 0 to 10 seconds.
# Use np.random.default_rng with a seed = last 4 digits of your Drexel ID.

import numpy as np
from typing import Tuple


def generate_data(seed: int,
                  n: int = 200,
                  mean_a: float = 25.0,
                  std_a: float = 3.0,
                  mean_b: float = 27.0,
                  std_b: float = 4.5,
                  t_min: float = 0.0,
                  t_max: float = 10.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic sensor temperature data.

    Parameters
    ----------
    seed : int
        Seed for numpy.random.default_rng to ensure reproducible output.
    n : int, optional
        Number of samples per sensor. Default is 200.
    mean_a : float, optional
        Mean temperature for Sensor A in degrees Celsius (default 25.0).
    std_a : float, optional
        Standard deviation for Sensor A (default 3.0).
    mean_b : float, optional
        Mean temperature for Sensor B in degrees Celsius (default 27.0).
    std_b : float, optional
        Standard deviation for Sensor B (default 4.5).
    t_min : float, optional
        Minimum timestamp in seconds (default 0.0).
    t_max : float, optional
        Maximum timestamp in seconds (default 10.0).

    Returns
    -------
    timestamps : numpy.ndarray, shape (n,), dtype float64
        Sorted timestamps in seconds, sampled uniformly on [t_min, t_max].
    sensor_a : numpy.ndarray, shape (n,), dtype float64
        Sensor A temperature readings in degrees Celsius.
    sensor_b : numpy.ndarray, shape (n,), dtype float64
        Sensor B temperature readings in degrees Celsius.

    Notes
    -----
    - Uses the modern NumPy Generator API (numpy.random.default_rng) for reproducibility.
    - Timestamps are sorted and the readings are reordered to match the timestamps so
      returned arrays are aligned for time-series plotting.
    """
    rng = np.random.default_rng(seed)

    # Draw samples
    t = rng.uniform(t_min, t_max, size=n)
    a = rng.normal(loc=mean_a, scale=std_a, size=n)
    b = rng.normal(loc=mean_b, scale=std_b, size=n)

    # Sort by time and ensure float64 dtype
    order = np.argsort(t)
    timestamps = t[order].astype(np.float64)
    sensor_a = a[order].astype(np.float64)
    sensor_b = b[order].astype(np.float64)

    return timestamps, sensor_a, sensor_b


if __name__ == '__main__':
    # Simple sanity check
    ts, a, b = generate_data(seed=3693)
    print('shapes:', ts.shape, a.shape, b.shape)
    print('means:', float(a.mean()), float(b.mean()))
