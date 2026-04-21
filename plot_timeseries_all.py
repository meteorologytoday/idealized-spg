import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

# Define the region: Subpolar Gyre (roughly)
lat_min, lat_max = 50, 65
lon_min, lon_max = -60, -10

base_path = 'data/data'
variables = {
    r'SST ($^\circ$C)': ('sea_surface_temperature', 'sosstsst'),
    'SSS (psu)': ('sea_surface_salinity', 'sosaline'),
    'MLD (m)': ('mixed_layer_depth_0_01', 'somxl010'),
    r'Heat Flux (W/m$^2$)': ('net_downward_heat_flux', 'sohefldo'),
    r'Freshwater Flux (kg/m$^2$/s)': ('net_upward_water_flux', 'sowaflup')
}

def get_timeseries(var_dir, var_name):
    files = sorted(glob.glob(f'{base_path}/{var_dir}/*.nc'))
    if not files:
        print(f"No files found for {var_name}")
        return None, None
    
    ds = xr.open_mfdataset(files, combine='nested', concat_dim='time_counter')
    
    # Define mask
    mask = (ds.nav_lat >= lat_min) & (ds.nav_lat <= lat_max) & \
           (ds.nav_lon >= lon_min) & (ds.nav_lon <= lon_max)
    
    # Calculate area-weighted mean if possible, but simple mean is usually okay for these grids
    ts = ds[var_name].where(mask).mean(dim=['y', 'x']).compute()
    
    return ds.time_counter.values, ts.values

def plot_timeseries():
    fig, axes = plt.subplots(len(variables), 1, figsize=(12, 12), sharex=True)
    
    for i, (label, (var_dir, var_name)) in enumerate(variables.items()):
        print(f"Processing {label}...")
        times, values = get_timeseries(var_dir, var_name)
        if times is not None:
            axes[i].plot(times, values)
            axes[i].set_ylabel(label)
            axes[i].grid(True, linestyle='--', alpha=0.7)
            if i == 0:
                axes[i].set_title(f'SPG Averaged Timeseries ({lat_min}N-{lat_max}N, {lon_min}W-{lon_max}W)')

    axes[-1].set_xlabel('Year')
    plt.tight_layout()
    plt.savefig('oras5_spg_timeseries_all.png', dpi=150)
    print("Saved oras5_spg_timeseries_all.png")

if __name__ == "__main__":
    plot_timeseries()
