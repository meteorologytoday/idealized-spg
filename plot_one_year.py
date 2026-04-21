import xarray as xr
import matplotlib as mplt
mplt.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import glob
import os

# Set base path
base_path = 'data/data'

def load_and_mean_djf(var_dir, var_name, years):
    """
    Loads data for a list of years and calculates the multi-year DJF mean.
    """
    if isinstance(years, int):
        years = [years]
    
    all_files = []
    for year in years:
        files = glob.glob(f'{base_path}/{var_dir}/*{year}*.nc')
        all_files.extend(files)
    
    all_files = sorted(all_files)
    
    if not all_files:
        print(f"No files found for {var_name} in years {years}")
        return None, None, None
    
    ds = xr.open_mfdataset(all_files, combine='nested', concat_dim='time_counter')
    ds = ds.rename({'time_counter': 'time'})
    
    # Select DJF season across all files and compute the mean
    djf_mean = ds[var_name].sel(time=ds.time.dt.season == 'DJF').mean(dim='time').compute()
    
    return djf_mean, ds.nav_lon, ds.nav_lat

def plot_one_year(years=2010, lon_range=(-70, -10), lat_range=(40, 70)):
    """
    Plots the average DJF state for the specified years and region.
    - years: int or list of ints
    - lon_range: tuple (min_lon, max_lon)
    - lat_range: tuple (min_lat, max_lat)
    """
    if isinstance(years, (int, str)):
        years_list = [int(years)]
    else:
        years_list = sorted(years)

    plt.rcParams.update({'font.size': 10})
    
    year_label = f"{years_list[0]}" if len(years_list) == 1 else f"{years_list[0]}-{years_list[-1]}"
    print(f"Loading and averaging data for DJF {year_label}...")
    
    # Load multi-year averages
    mld, lon, lat = load_and_mean_djf('mixed_layer_depth_0_01', 'somxl010', years_list)
    if mld is None:
        return
        
    sst, _, _ = load_and_mean_djf('sea_surface_temperature', 'sosstsst', years_list)
    sss, _, _ = load_and_mean_djf('sea_surface_salinity', 'sosaline', years_list)
    hflux, _, _ = load_and_mean_djf('net_downward_heat_flux', 'sohefldo', years_list)
    wflux, _, _ = load_and_mean_djf('net_upward_water_flux', 'sowaflup', years_list)
    
    # Wind stress components
    taux, _, _ = load_and_mean_djf('zonal_wind_stress', 'sozotaux', years_list)
    tauy, _, _ = load_and_mean_djf('meridional_wind_stress', 'sometauy', years_list)
    
    # Convert freshwater flux to mm/day
    if wflux is not None:
        wflux = wflux * 86400

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), 
                             subplot_kw={'projection': ccrs.PlateCarree()})
    axes = axes.flatten()
    
    # Variables configuration
    plot_config = [
        (mld, 'viridis_r', 'MLD (m)', 0, 1000),
        (sst, 'RdYlBu_r', r'SST ($^\circ$C)', 0, 15),
        (sss, 'YlGnBu', 'SSS (psu)', 34, 36),
        (hflux, 'RdBu_r', r'Heat Flux (W/m$^2$)', -300, 300),
        (wflux, 'BrBG', 'Freshwater Flux (mm/day)', -5, 5),
    ]

    for i in range(5):
        data, cmap, title, vmin, vmax = plot_config[i]
        ax = axes[i]
        ax.set_extent([lon_range[0], lon_range[1], lat_range[0], lat_range[1]], crs=ccrs.PlateCarree())

        if data is not None:
            im = ax.pcolormesh(lon, lat, data, transform=ccrs.PlateCarree(), 
                               cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')

            # Use explicit ticks to ensure colorbar consistency
            ticks = np.linspace(vmin, vmax, 5)
            plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.08, shrink=0.8, 
                         ticks=ticks, extend='both')

        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.LAND, zorder=10, facecolor='lightgrey')
        ax.set_title(title, fontweight='bold')

        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
    # Panel 6: Wind Stress Vectors
    ax = axes[5]
    ax.set_extent([lon_range[0], lon_range[1], lat_range[0], lat_range[1]], crs=ccrs.PlateCarree())
    
    if taux is not None and tauy is not None:
        skip = 5
        q = ax.quiver(lon[::skip, ::skip], lat[::skip, ::skip], 
                      taux[::skip, ::skip], tauy[::skip, ::skip], 
                      transform=ccrs.PlateCarree(), scale=2)
        ax.quiverkey(q, 0.9, 0.1, 0.2, r'0.2 N/m$^2$', labelpos='E', coordinates='figure')
    
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND, zorder=10, facecolor='lightgrey')
    ax.set_title('Wind Stress (Vector)', fontweight='bold')
    
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

    plt.suptitle(f'ORAS5 DJF Mean ({year_label}) - SPG Region', fontsize=22, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    os.makedirs('figures', exist_ok=True)
    suffix = f"{years_list[0]}_{years_list[-1]}" if len(years_list) > 1 else f"{years_list[0]}"
    output_path = f'figures/oras5_djf_avg_{suffix}_6panels.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Saved {output_path}")
    plt.close()

if __name__ == "__main__":
    years_to_plot = list(range(2003, 2015))

    for year in years_to_plot:
        plot_one_year(years=year, lon_range=(-70, -10), lat_range=(40, 70))
    plot_one_year(years=years_to_plot,  lon_range=(-70, -10), lat_range=(40, 70))
