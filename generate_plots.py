# Intent: Generate synthetic sensor data for lab3 plots
# Sensor A: mean 25 C, std 3 C, 200 readings.
# Sensor B: mean 27 C, std 4.5 C, 200 readings.
# Also generate 200 timestamps uniformly from 0 to 10 seconds.
# Use np.random.default_rng with a seed = last 4 digits of your Drexel ID.

import numpy as np
import matplotlib.pyplot as plt
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


def plot_scatter(ax,
                 timestamps: np.ndarray,
                 sensor_a: np.ndarray,
                 sensor_b: np.ndarray,
                 *,
                 label_a: str = 'Sensor A',
                 label_b: str = 'Sensor B',
                 color_a: str = 'tab:blue',
                 color_b: str = 'tab:orange',
                 markersize: float = 30.0,
                 alpha: float = 0.7,
                 title: str | None = None) -> None:
    """Plot scatter time-series of two sensors on an existing Axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The Axes object to plot onto. Modified in place.
    timestamps : numpy.ndarray, shape (n,)
        Time values in seconds (float64).
    sensor_a : numpy.ndarray, shape (n,)
        Temperature readings for Sensor A (float64).
    sensor_b : numpy.ndarray, shape (n,)
        Temperature readings for Sensor B (float64).
    label_a : str, optional
        Legend label for Sensor A. Default 'Sensor A'.
    label_b : str, optional
        Legend label for Sensor B. Default 'Sensor B'.
    color_a : str, optional
        Color for Sensor A markers. Default 'tab:blue'.
    color_b : str, optional
        Color for Sensor B markers. Default 'tab:orange'.
    markersize : float, optional
        Marker size for scatter points. Default 30.0.
    alpha : float, optional
        Marker alpha (transparency). Default 0.7.
    title : str or None, optional
        Axes title. If None a sensible default is used.

    Returns
    -------
    None
        The function modifies ``ax`` in place and returns nothing.

    Notes
    -----
    - Computes and annotates the Pearson correlation coefficient between
      the two sensor readings (across the provided samples) in the title.
    - Does not create a new figure; caller is responsible for figure management.
    """
    # Basic input validation / normalization
    timestamps = np.asarray(timestamps, dtype=np.float64)
    sensor_a = np.asarray(sensor_a, dtype=np.float64)
    sensor_b = np.asarray(sensor_b, dtype=np.float64)

    # Scatter plots for both sensors vs time
    ax.scatter(timestamps, sensor_a, s=markersize, c=color_a, alpha=alpha, label=label_a, edgecolors='w', linewidths=0.3)
    ax.scatter(timestamps, sensor_b, s=markersize, c=color_b, alpha=alpha, label=label_b, edgecolors='w', linewidths=0.3)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Temperature (°C)')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()

    # Compute Pearson correlation coefficient for annotation
    try:
        r = np.corrcoef(sensor_a, sensor_b)[0, 1]
    except Exception:
        r = np.nan

    if title is None:
        title = f'Time-series scatter (r = {r:.2f})' if np.isfinite(r) else 'Time-series scatter'
    ax.set_title(title)


if __name__ == '__main__':
    # Simple sanity check
    ts, a, b = generate_data(seed=3693)
    print('shapes:', ts.shape, a.shape, b.shape)
    print('means:', float(a.mean()), float(b.mean()))

    # Demonstrate plotting function by creating and saving a figure
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_scatter(ax, ts, a, b)
    fig.tight_layout()
    fig.savefig('scatter_demo.png')
    print("Saved 'scatter_demo.png'")

# Create plot_scatter(sensor_a, sensor_b, timestamps, ax) that draws
# the scatter plot from the notebook onto the given Axes object.
# NumPy-style docstring. Modifies ax in place, returns None.