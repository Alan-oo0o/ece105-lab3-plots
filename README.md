# Lab 3 Sensor Plots

A small utility to generate synthetic sensor temperature data and produce exploratory plots (time-series scatter, overlaid histogram, and side-by-side boxplot).

## Installation

1. Activate the ece105 conda environment:

   conda activate ece105

2. Install required packages (using conda or mamba is recommended):

   conda install numpy matplotlib

   or

   mamba install numpy matplotlib

(Alternatively, you can pip install these packages inside the activated environment: `pip install numpy matplotlib`.)

## Usage

Run the script to generate reproducible synthetic data and save a consolidated figure:

    python generate_plots.py

You can adjust the RNG seed by editing the default in the script or by calling the functions from Python directly.

## Example output

Running the script produces a single PNG file named `sensor_analysis.png` containing a **2x2 figure** with:

- Top-Left: Time vs Temperature scatter plot for Sensor A and Sensor B
- Top-Right: Overlaid histogram comparing the temperature distributions
- Bottom-Left: Side-by-side boxplot comparison with overall mean indicated
- Bottom-Right: Empty / placeholder for summary statistics

## AI tools used and disclosure

I used GitHub Copilot to assist with this lab. I followed the explain first pattern and context scoping techniques. I used intent comments for each function to specify requirements prior to allowing Copilot to suggest implemnetations to ensure code generated adhered to specific requirements. 
