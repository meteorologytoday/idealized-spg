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

def plot_structure():
    # Set global font size for better readability
    plt.rcParams.update({'font.size': 14})
    
    print("Loading MLD data...")
    mld_files = sorted(glob.glob(f'{base_path}/mixed_layer_depth_0_01/*.nc'))
    
    if not mld_files:
        print(f"No files found in {base_path}/mixed_layer_depth_0_01/")
        return

    # Open all files
    ds = xr.open_mfdataset(mld_files, combine='nested', concat_dim='time_counter')
    
    # Rename time_counter to time for convenience with groupby
    ds = ds.rename({'time_counter': 'time'})
    
    # Group by season and calculate mean and std
    print("Calculating seasonal statistics...")
    seasonal_mean = ds.somxl010.groupby('time.season').mean(dim='time').compute()
    seasonal_std = ds.somxl010.groupby('time.season').std(dim='time').compute()
    
    # Extract year range for suptitle
    start_year = ds.time.dt.year.min().values
    end_year = ds.time.dt.year.max().values
    
    seasons = ['DJF', 'MAM', 'JJA', 'SON']
    
    # Define the SPG region for mapping
    lon_range = [-80, 20]
    lat_range = [30, 80]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), 
                             subplot_kw={'projection': ccrs.PlateCarree()})
    axes = axes.flatten()
    
    # Add suptitle with year range
    fig.suptitle(f'ORAS5 MLD Seasonal Statistics ({start_year}-{end_year})', 
                 fontsize=22, fontweight='bold', y=0.95)
    
    for i, season in enumerate(seasons):
        ax = axes[i]
        ax.set_extent([lon_range[0], lon_range[1], lat_range[0], lat_range[1]], 
                      crs=ccrs.PlateCarree())
        
        # Shading: Mean field (Range [0, 1000])
        data_mean = seasonal_mean.sel(season=season)
        im = ax.pcolormesh(ds.nav_lon, ds.nav_lat, data_mean,
                           transform=ccrs.PlateCarree(), cmap='viridis_r', 
                           vmin=0, vmax=1000, shading='auto')
        
        # Contours: 1 standard deviation (Thicker lines)
        data_std = seasonal_std.sel(season=season)
        cs = ax.contour(ds.nav_lon, ds.nav_lat, data_std,
                        transform=ccrs.PlateCarree(), colors='black', 
                        levels=[200, 500], alpha=0.8, linewidths=1.5)
        ax.clabel(cs, fontsize=12, fmt='%1.0f')

        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.LAND, zorder=10, facecolor='lightgrey')
        ax.set_title(f'MLD Seasonal Mean & STD - {season}', fontsize=16, fontweight='bold')

        # Add gridlines and lat/lon labels
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', 
                          alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 12}
        gl.ylabel_style = {'size': 12}
        
    # Add colorbar with larger labels
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Mixed Layer Depth (m)', fontsize=16)
    
    os.makedirs('figures', exist_ok=True)
    output_path = 'figures/oras5_spg_structure.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Saved {output_path}")

if __name__ == "__main__":
    plot_structure()
