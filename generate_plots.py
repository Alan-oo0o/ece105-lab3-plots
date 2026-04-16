import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

"""
Module for generating synthetic sensor data and visualizing it through 
scatter plots, histograms, and boxplots.
"""

# Intent: Generate synthetic sensor data for lab3 plots
# Sensor A: mean 25 C, std 3 C, 200 readings.
# Sensor B: mean 27 C, std 4.5 C, 200 readings.
# Also generate 200 timestamps uniformly from 0 to 10 seconds.
# Use np.random.default_rng with a seed = last 4 digits of your Drexel ID.

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

# Create plot_scatter(sensor_a, sensor_b, timestamps, ax) that draws
# the scatter plot from the notebook onto the given Axes object.
# NumPy-style docstring. Modifies ax in place, returns None.

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

# Overlaid histogram of Sensor A and Sensor B temperature distributions.
# Use 30 bins, alpha=0.5 for transparency so both distributions are visible.
# Add vertical dashed lines at each sensor's mean.
# Include a legend labeling each sensor.

def plot_histogram(ax,
                   sensor_a: np.ndarray,
                   sensor_b: np.ndarray,
                   *,
                   bins: int = 30,
                   label_a: str = 'Sensor A',
                   label_b: str = 'Sensor B',
                   color_a: str = 'tab:blue',
                   color_b: str = 'tab:orange',
                   alpha: float = 0.5,
                   density: bool = False,
                   show_means: bool = True,
                   title: str | None = None) -> None:
    """Draw overlaid histograms of two sensor distributions on an Axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw the histograms onto (modified in place).
    sensor_a : numpy.ndarray, shape (n,)
        Samples for Sensor A (float64).
    sensor_b : numpy.ndarray, shape (n,)
        Samples for Sensor B (float64).
    bins : int, optional
        Number of histogram bins (default 30).
    label_a, label_b : str, optional
        Legend labels for each sensor.
    color_a, color_b : str, optional
        Colors for the histogram bars.
    alpha : float, optional
        Transparency for histogram bars (default 0.5).
    density : bool, optional
        If True, normalize histograms to form a probability density.
    show_means : bool, optional
        If True, draw vertical dashed lines at each sensor's mean.
    title : str or None, optional
        Axes title. If None a sensible default is used.

    Returns
    -------
    None
        Modifies ``ax`` in place and returns nothing.

    Notes
    -----
    - Uses semi-transparent, overlaid histograms so both distributions are visible.
    - Adds a legend and grid for readability.
    """
    sensor_a = np.asarray(sensor_a, dtype=np.float64)
    sensor_b = np.asarray(sensor_b, dtype=np.float64)

    ax.hist(sensor_a, bins=bins, color=color_a, alpha=alpha, label=label_a, density=density)
    ax.hist(sensor_b, bins=bins, color=color_b, alpha=alpha, label=label_b, density=density)

    if show_means:
        ax.axvline(float(sensor_a.mean()), color=color_a, linestyle='--', linewidth=1.25, label=f"{label_a} mean")
        ax.axvline(float(sensor_b.mean()), color=color_b, linestyle='--', linewidth=1.25, label=f"{label_b} mean")

    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Count' if not density else 'Density')
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend()
    if title is None:
        title = f'Histogram of {label_a} and {label_b}'
    ax.set_title(title)

# Side-by-side box plot comparing Sensor A and Sensor B distributions.
# Label x-axis with sensor names, y-axis with "Temperature (deg C)".
# Add a horizontal dashed line at the overall mean of both sensors combined.

def plot_boxplot(ax,
                 sensor_a: np.ndarray,
                 sensor_b: np.ndarray,
                 *,
                 labels: tuple | None = None,
                 colors: tuple | None = None,
                 showmeans: bool = False,
                 boxprops: dict | None = None,
                 medianprops: dict | None = None,
                 title: str | None = None) -> None:
    """Draw a side-by-side boxplot comparing two sensor distributions.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw the boxplot onto (modified in place).
    sensor_a : numpy.ndarray, shape (n,)
        Samples for Sensor A (float64).
    sensor_b : numpy.ndarray, shape (n,)
        Samples for Sensor B (float64).
    labels : tuple of str, optional
        Labels for the x-axis categories (default ('Sensor A', 'Sensor B')).
    colors : tuple of str, optional
        Fill colors for the boxes (default ('tab:blue','tab:orange')).
    showmeans : bool, optional
        If True, show the mean marker for each box (default False).
    boxprops, medianprops : dict, optional
        Additional properties forwarded to matplotlib's boxplot for boxes/median.
    title : str or None, optional
        Axes title. If None a sensible default is used.

    Returns
    -------
    None
        Modifies ``ax`` in place and returns nothing.

    Notes
    -----
    - Boxes are colored using ``patch_artist=True`` so fill colors can be set.
    - Draws a horizontal dashed line at the overall mean of both sensors combined.
    """
    sensor_a = np.asarray(sensor_a, dtype=np.float64)
    sensor_b = np.asarray(sensor_b, dtype=np.float64)

    if labels is None:
        labels = ('Sensor A', 'Sensor B')
    if colors is None:
        colors = ('tab:blue', 'tab:orange')

    bp = ax.boxplot([sensor_a, sensor_b], patch_artist=True, labels=labels, showmeans=showmeans,
                    boxprops=boxprops or {}, medianprops=medianprops or {})

    # Color the boxes
    for patch, color in zip(bp.get('boxes', []), colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Style median lines if available
    if 'medians' in bp:
        for med in bp['medians']:
            med.set(color='black', linewidth=1.0)

    overall_mean = float(np.concatenate([sensor_a, sensor_b]).mean())
    ax.axhline(overall_mean, color='gray', linestyle='--', linewidth=1.0, label=f'Overall mean ({overall_mean:.2f}°C)')

    ax.set_ylabel('Temperature (°C)')
    ax.grid(True, axis='y', linestyle=':', alpha=0.5)
    ax.legend()
    if title is None:
        title = 'Boxplot comparison'
    ax.set_title(title)


# Create main() that generates data, creates a 1x3 subplot figure,
# calls each plot function, adjusts layout, and saves as sensor_analysis.png
# at 150 DPI with tight bounding box.

def main(seed: int = 3693) -> None:
    """Generate data and produce a consolidated 1x3 multi-panel figure.

    Parameters
    ----------
    seed : int
        Random seed passed to :func:`generate_data` for reproducibility. Default is 3693.

    Returns
    -------
    None
        Creates and saves a single PNG file in the current working directory:
        - 'sensor_analysis.png' : 1x3 figure with Time vs Temperature (left),
          overlaid histogram (middle), and boxplot comparison (right).

    Notes
    -----
    - Uses the helper plotting functions in this module (plot_scatter,
      plot_histogram, plot_boxplot) which modify Axes in place.
    - The function closes the figure after saving to avoid consuming memory.
    - The scatter panel shows only Time vs Temperature for both sensors; the
      redundant A vs B panel and correlation calculation were removed.
    """
    ts, a, b = generate_data(seed=seed)

    # Arrange plots in a 2x2 subplot grid
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    
    # Left: time-series scatter for both sensors
    plot_scatter(axs[0, 0], ts, a, b, title='Time vs Temperature')
    
    # Middle: histogram
    plot_histogram(axs[0, 1], a, b, title='Histogram of temperatures')

    # Right: boxplot
    plot_boxplot(axs[1, 0], a, b, title='Boxplot comparison')

    # Note: axs[1, 1] remains empty as per lab instructions.

    fig.tight_layout()
    out_fname = 'sensor_analysis.png'
    fig.savefig(out_fname, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved: {out_fname}")


if __name__ == '__main__':
    main()

