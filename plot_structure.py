import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import glob
import os

# Set base path
base_path = 'data/data'

def plot_structure():
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
    
    seasons = ['DJF', 'MAM', 'JJA', 'SON']
    
    # Define the SPG region for mapping
    lon_range = [-80, 0]
    lat_range = [30, 80]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), 
                             subplot_kw={'projection': ccrs.PlateCarree()})
    axes = axes.flatten()
    
    for i, season in enumerate(seasons):
        ax = axes[i]
        ax.set_extent([lon_range[0], lon_range[1], lat_range[0], lat_range[1]], 
                      crs=ccrs.PlateCarree())
        
        # Shading: Mean field
        data_mean = seasonal_mean.sel(season=season)
        im = ax.pcolormesh(ds.nav_lon, ds.nav_lat, data_mean,
                           transform=ccrs.PlateCarree(), cmap='viridis_r', 
                           vmax=1500, shading='auto')
        
        # Contours: 1 standard deviation
        data_std = seasonal_std.sel(season=season)
        ax.contour(ds.nav_lon, ds.nav_lat, data_std,
                        transform=ccrs.PlateCarree(), colors='white', 
                        levels=[200, 500], alpha=0.7, linewidths=0.5)
        
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.LAND, zorder=10, facecolor='lightgrey')
        ax.set_title(f'MLD Seasonal Mean & STD - {season}')
        
    # Add colorbar
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    cbar_ax = fig.add_axes([0.15, 0.04, 0.7, 0.02])
    fig.colorbar(im, cax=cbar_ax, orientation='horizontal', label='Mixed Layer Depth (m)')
    
    plt.savefig('oras5_spg_structure.png', dpi=150)
    print("Saved oras5_spg_structure.png")

if __name__ == "__main__":
    plot_structure()
