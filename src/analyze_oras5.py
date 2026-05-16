import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from datetime import datetime

# Define the region: Subpolar Gyre (roughly)
lat_min, lat_max = 50, 65
lon_min, lon_max = -60, -10

base_path = 'PIK_2026_backup/SPG/data'
variables = {
    'SST': ('sea_surface_temperature', 'sosstsst'),
    'MLD': ('mixed_layer_depth_0_01', 'somxl010'),
    'TauX': ('zonal_wind_stress', 'sozotaux'),
    'TauY': ('meridional_wind_stress', 'sometauy'),
    'HeatFlux': ('net_downward_heat_flux', 'sohefldo')
}

def get_timeseries(var_dir, var_name):
    files = sorted(glob.glob(f'{base_path}/{var_dir}/*.nc'))
    times = []
    values = []
    
    # Load first file to get mask
    ds0 = xr.open_dataset(files[0])
    mask = (ds0.nav_lat >= lat_min) & (ds0.nav_lat <= lat_max) & \
           (ds0.nav_lon >= lon_min) & (ds0.nav_lon <= lon_max)
    
    for f in files:
        ds = xr.open_dataset(f)
        # Assuming time_counter is the time dimension
        val = ds[var_name].where(mask).mean(dim=['y', 'x']).values[0]
        t = ds.time_counter.values[0]
        times.append(t)
        values.append(val)
        ds.close()
    
    return np.array(times), np.array(values)

fig, axes = plt.subplots(5, 1, figsize=(12, 15), sharex=True)

for i, (label, (var_dir, var_name)) in enumerate(variables.items()):
    print(f"Processing {label}...")
    times, values = get_timeseries(var_dir, var_name)
    axes[i].plot(times, values)
    axes[i].set_ylabel(label)
    axes[i].grid(True)

axes[-1].set_xlabel('Time')
plt.tight_layout()
plt.savefig('oras5_spg_timeseries.png')
print("Saved oras5_spg_timeseries.png")
