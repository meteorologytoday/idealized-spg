# Project: Subpolar Gyre Analysis

This project focuses on analyzing ocean reanalysis data in the North Atlantic Subpolar Gyre (SPG) region.

## Personal request

You are a senior engineer. You need to be strict on the code quality and flexibility. Do not produce code that may result in 
unexpected or unportable behaviors. We are equal, and we talk nice :)

## Restriction

You can only modify files within this project.

## Git

- repo URL: https://github.com/meteorologytoday/idealized-spg.git

## Project Guidelines

- **Environment:** Always use the `jaxesm` conda environment for data processing and visualization.
  - Required libraries: `xarray`, `numpy`, `matplotlib`, `cartopy`.
- **Data Source:** The primary data is located at `data`.
  - Data is organized into subdirectories by variable (e.g., `sea_surface_temperature`, `mixed_layer_depth_0_01`).
  - Files are monthly NetCDF files with `nav_lat`, `nav_lon`, and `time_counter` dimensions.
- **Region of Interest:** The Subpolar Gyre (SPG), approximately bounded by:
  - Latitude: 50°N to 65°N
  - Longitude: 60°W to 10°W
- **Conventions:**
  - Use `xarray` for all NetCDF data handling.
  - Use `cartopy` for spatial mapping to ensure proper geographic projections.
  - Maintain scripts for reproducibility (e.g., `analyze_oras5.py`, `plot_structure.py`).

## Reanalysis Dataset

- ORAS5
   1. `sosstsst`: Sea Surface Temperature (C)
   2. `somxl010`: Mixed Layer Depth (m) - 0.01 density criterion
   3. `sozotaux`: Zonal Wind Stress (N/m^2)
   4. `sometauy`: Meridional Wind Stress (N/m^2)
   5. `sohefldo`: Net Downward Heat Flux (W/m^2)
- GLORYS 1/12 degrees


## Code Environment

- Use `conda activate jaxesm` for python.

## Description of Plotting Files

### `plot_structure.py`
 
    1. It plots the 2D map of fields (mixed layer depth) of grouped by seasons (DJF, MAM, JJA, SON).
    2. Shading plots the mean field, with contours 1 standard deviation.

### `plot_timeseries_all.py`

    1. It plots the timeseries of averaged values over a selected region (adjustable latitude-longitude box).
    2. Variables include: SST, SSS, mixed layer depth, downward heat flux, freshwater flux.

### `plot_timeseries_monthly.py`

    1. Same as `plot_timeseries_all.py`, but the time axis is month. 
    2. Data plotted are the mean of that month of all years, with 1 standard deviations plotted as the error bars.


