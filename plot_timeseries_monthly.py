import xarray as xr
import matplotlib as mplt
mplt.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

base_path = 'data/data'
variables = {
    r'SST ($^\circ$C)': ('sea_surface_temperature', 'sosstsst'),
    'SSS (psu)': ('sea_surface_salinity', 'sosaline'),
    'MLD (m)': ('mixed_layer_depth_0_01', 'somxl010'),
    r'Heat Flux (W/m$^2$)': ('net_downward_heat_flux', 'sohefldo'),
    r'Freshwater Flux (mm/day)': ('net_upward_water_flux', 'sowaflup'),
    r'Wind Stress (N/m$^2$)': ('wind_stress', 'wind_stress')
}

def get_monthly_climatology(var_dir, var_name, lat_range, lon_range):
    if var_name == 'wind_stress':
        files_u = sorted(glob.glob(f'{base_path}/zonal_wind_stress/*.nc'))
        files_v = sorted(glob.glob(f'{base_path}/meridional_wind_stress/*.nc'))
        if not files_u or not files_v:
            return None, None
        
        ds_u = xr.open_mfdataset(files_u, combine='nested', concat_dim='time_counter').rename({'time_counter': 'time'})
        ds_v = xr.open_mfdataset(files_v, combine='nested', concat_dim='time_counter').rename({'time_counter': 'time'})
        
        data = np.sqrt(ds_u.sozotaux**2 + ds_v.sometauy**2)
        nav_lat, nav_lon = ds_u.nav_lat, ds_u.nav_lon
    else:
        files = sorted(glob.glob(f'{base_path}/{var_dir}/*.nc'))
        if not files:
            print(f"No files found for {var_name}")
            return None, None
            
        ds = xr.open_mfdataset(files, combine='nested', concat_dim='time_counter')
        ds = ds.rename({'time_counter': 'time'})
        data = ds[var_name]
        nav_lat, nav_lon = ds.nav_lat, ds.nav_lon
    
    mask = (nav_lat >= lat_range[0]) & (nav_lat <= lat_range[1]) & \
           (nav_lon >= lon_range[0]) & (nav_lon <= lon_range[1])
    
    # First spatial mean
    spatial_mean = data.where(mask).mean(dim=['y', 'x'])
    
    # Convert freshwater flux to mm/day if applicable
    if var_name == 'sowaflup':
        spatial_mean = spatial_mean * 86400
    
    # Then group by month
    monthly_mean = spatial_mean.groupby('time.month').mean(dim='time').compute()
    monthly_std = spatial_mean.groupby('time.month').std(dim='time').compute()
    
    return monthly_mean, monthly_std

def format_coord(val, coord_type='lat'):
    """Format lat/lon values with N/S and E/W suffixes."""
    if coord_type == 'lat':
        suffix = 'N' if val >= 0 else 'S'
    else:
        suffix = 'E' if val >= 0 else 'W'
    return f"{abs(val):g}{suffix}"

def plot_timeseries_monthly(lat_range=(50, 65), lon_range=(-60, -10)):
    fig, axes = plt.subplots(len(variables), 1, figsize=(10, 16), sharex=True)
    months = np.arange(1, 13)
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # Format range strings for the title
    lat_str = f"{format_coord(lat_range[0], 'lat')}-{format_coord(lat_range[1], 'lat')}"
    lon_str = f"{format_coord(lon_range[0], 'lon')}-{format_coord(lon_range[1], 'lon')}"

    for i, (label, (var_dir, var_name)) in enumerate(variables.items()):
        print(f"Processing {label}...")
        m_mean, m_std = get_monthly_climatology(var_dir, var_name, lat_range, lon_range)

        if m_mean is not None:
            axes[i].errorbar(months, m_mean, yerr=m_std, fmt='-o', capsize=5, label=r'Mean $\pm$ 1 STD')
            axes[i].set_ylabel(label, fontsize=12)
            axes[i].grid(True, linestyle='--', alpha=0.7)
            axes[i].set_xticks(months)
            axes[i].set_xticklabels(month_labels)

            if i == 0:
                axes[i].set_title(f'Monthly Climatology ({lat_str}, {lon_str})', 
                                  fontsize=14, fontweight='bold')

    axes[-1].set_xlabel('Month', fontsize=12)
    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    output_path = 'figures/oras5_spg_timeseries_monthly.png'
    plt.savefig(output_path, dpi=150)
    print(f"Saved {output_path}")

if __name__ == "__main__":
    # Example: SPG region
    plot_timeseries_monthly(lat_range=(60-2.5, 60+2.5), lon_range=(-55-2.5, -55+2.5))
