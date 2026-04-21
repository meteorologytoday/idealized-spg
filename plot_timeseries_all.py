import xarray as xr
import matplotlib as mplt
mplt.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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

def get_timeseries(var_dir, var_name, lat_range, lon_range):
    if var_name == 'wind_stress':
        # Load both components for wind stress wind_stress
        files_u = sorted(glob.glob(f'{base_path}/zonal_wind_stress/*.nc'))
        files_v = sorted(glob.glob(f'{base_path}/meridional_wind_stress/*.nc'))
        if not files_u or not files_v:
            return None, None
        
        ds_u = xr.open_mfdataset(files_u, combine='nested', concat_dim='time_counter')
        ds_v = xr.open_mfdataset(files_v, combine='nested', concat_dim='time_counter')
        
        # Calculate wind_stress
        data = np.sqrt(ds_u.sozotaux**2 + ds_v.sometauy**2)
        time = ds_u.time_counter.values
        nav_lat, nav_lon = ds_u.nav_lat, ds_u.nav_lon
    else:
        files = sorted(glob.glob(f'{base_path}/{var_dir}/*.nc'))
        if not files:
            print(f"No files found for {var_name}")
            return None, None
        
        ds = xr.open_mfdataset(files, combine='nested', concat_dim='time_counter')
        data = ds[var_name]
        time = ds.time_counter.values
        nav_lat, nav_lon = ds.nav_lat, ds.nav_lon

    # Define mask
    mask = (nav_lat >= lat_range[0]) & (nav_lat <= lat_range[1]) & \
           (nav_lon >= lon_range[0]) & (nav_lon <= lon_range[1])
    
    # Calculate simple mean over the mask
    ts = data.where(mask).mean(dim=['y', 'x']).compute()
    
    # Convert freshwater flux to mm/day if applicable
    if var_name == 'sowaflup':
        ts = ts * 86400
    
    return time, ts.values

def format_coord(val, coord_type='lat'):
    """Format lat/lon values with N/S and E/W suffixes."""
    if coord_type == 'lat':
        suffix = 'N' if val >= 0 else 'S'
    else:
        suffix = 'E' if val >= 0 else 'W'
    return f"{abs(val):g}{suffix}"

def plot_timeseries_all(lat_range=(50, 65), lon_range=(-60, -10)):
    fig, axes = plt.subplots(len(variables), 1, figsize=(12, 14), sharex=True)

    # Format range strings for the title
    lat_str = f"{format_coord(lat_range[0], 'lat')}-{format_coord(lat_range[1], 'lat')}"
    lon_str = f"{format_coord(lon_range[0], 'lon')}-{format_coord(lon_range[1], 'lon')}"

    for i, (label, (var_dir, var_name)) in enumerate(variables.items()):
        print(f"Processing {label}...")
        times, values = get_timeseries(var_dir, var_name, lat_range, lon_range)
        if times is not None:
            axes[i].plot(times, values)
            axes[i].set_ylabel(label, fontsize=12)
            axes[i].grid(True, linestyle='--', alpha=0.7)
            if i == 0:
                axes[i].set_title(f'Averaged Timeseries ({lat_str}, {lon_str})', 
                                  fontsize=14, fontweight='bold')

    axes[-1].set_xlabel('Year', fontsize=12)
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(axes[-1].get_xticklabels(), rotation=45)
    
    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    output_path = 'figures/oras5_spg_timeseries_all.png'
    plt.savefig(output_path, dpi=150)
    print(f"Saved {output_path}")

if __name__ == "__main__":
    # Example: SPG region
    plot_timeseries_all(lat_range=(60-2.5, 60+2.5), lon_range=(-55-2.5, -55+2.5))
