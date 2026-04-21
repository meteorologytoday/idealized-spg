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

def get_monthly_climatology(var_dir, var_name):
    files = sorted(glob.glob(f'{base_path}/{var_dir}/*.nc'))
    if not files:
        print(f"No files found for {var_name}")
        return None, None
        
    ds = xr.open_mfdataset(files, combine='nested', concat_dim='time_counter')
    ds = ds.rename({'time_counter': 'time'})
    
    mask = (ds.nav_lat >= lat_min) & (ds.nav_lat <= lat_max) & \
           (ds.nav_lon >= lon_min) & (ds.nav_lon <= lon_max)
    
    # First spatial mean
    spatial_mean = ds[var_name].where(mask).mean(dim=['y', 'x'])
    
    # Then group by month
    monthly_mean = spatial_mean.groupby('time.month').mean(dim='time').compute()
    monthly_std = spatial_mean.groupby('time.month').std(dim='time').compute()
    
    return monthly_mean, monthly_std

def plot_monthly_climatology():
    fig, axes = plt.subplots(len(variables), 1, figsize=(10, 15), sharex=True)
    months = np.arange(1, 13)
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    for i, (label, (var_dir, var_name)) in enumerate(variables.items()):
        print(f"Processing {label}...")
        m_mean, m_std = get_monthly_climatology(var_dir, var_name)
        
        if m_mean is not None:
            axes[i].errorbar(months, m_mean, yerr=m_std, fmt='-o', capsize=5, label=r'Mean $\pm$ 1 STD')
            axes[i].set_ylabel(label)
            axes[i].grid(True, linestyle='--', alpha=0.7)
            axes[i].set_xticks(months)
            axes[i].set_xticklabels(month_labels)
            
            if i == 0:
                axes[i].set_title(f'SPG Monthly Climatology ({lat_min}N-{lat_max}N, {lon_min}W-{lon_max}W)')

    axes[-1].set_xlabel('Month')
    plt.tight_layout()
    plt.savefig('oras5_spg_timeseries_monthly.png', dpi=150)
    print("Saved oras5_spg_timeseries_monthly.png")

if __name__ == "__main__":
    plot_monthly_climatology()
