import os

import matplotlib as mplt
mplt.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

import data_loader as dl

# Fixed 2x4 panel grid (row 1 has 4 panels, row 2 has 3 plus one blank slot),
# addressed by flat index so every product places a given quantity in the same
# spot -- panels a product can't provide are hidden via _hide_panel() but keep
# their place, so the layout stays identical across products.
N_ROWS, N_COLS = 2, 4

# (panel index, standardized variable name, colormap, axis label, vmin, vmax)
SCALAR_PANELS = [
    (0, 'mld',        'viridis_r', 'MLD (m)',                   0,    1000),
    (1, 'sst',        'RdYlBu_r',  r'SST ($^\circ$C)',          0,    15),
    (2, 'sss',        'YlGnBu',    'SSS (psu)',                 34,   36),
    (4, 'heat_flux',  'RdBu_r',    r'Heat Flux (W/m$^2$)',      -300, 300),
    (5, 'water_flux', 'BrBG',      'Freshwater Flux (mm/day)',  -5,   5),
]

# Panel 3 (row 1, rightmost): the ocean's *own* surface currents, drawn as a
# streamline plot. This is a different physical quantity from wind stress
# (the atmosphere's forcing on the ocean surface, panel 6 below) and gets its
# own panel so the two are never conflated.
#
# matplotlib's streamplot needs a regular 1D lat/lon grid; GLORYS provides one
# natively, so it is the starting point here. ORAS5's curvilinear NEMO grid
# would need regridding first, so it is not yet wired up (panel stays hidden).
SURFACE_CURRENT_PANEL_INDEX = 3
SURFACE_CURRENT_PANELS = {
    'GLORYS': dict(u='uo', v='vo', title='Surface Current (Streamline)', cmap='viridis', label='Speed (m s$^{-1}$)'),
}

# Panel 6 (row 2): wind stress -- the atmospheric forcing on the ocean
# surface. Only ORAS5 publishes it among our surface fields.
WIND_STRESS_PANEL_INDEX = 6
WIND_STRESS_PANELS = {
    'ORAS5': dict(u='taux', v='tauy', title='Wind Stress (Vector)', key_value=0.2, key_label=r'0.2 N/m$^2$', scale=2, skip=5),
}

# Convert some fluxes to more readable units after loading.
_UNIT_CONVERSIONS = {
    'water_flux': lambda da: da * 86400,  # kg m^-2 s^-1 -> mm/day
}


def _load_djf_mean(dataset, variable, years):
    """
    Load `variable` for `dataset`/`years`, select DJF months, and average over
    time. Returns None if the product does not provide this variable.
    """
    da = dl.load_surface(dataset, variable, years=years)
    if da is None:
        return None

    convert = _UNIT_CONVERSIONS.get(variable)
    if convert is not None:
        da = convert(da)

    return da.sel(time=da.time.dt.season == 'DJF').mean(dim='time').compute()


def _set_map_extent(ax, lon_range, lat_range):
    ax.set_extent([lon_range[0], lon_range[1], lat_range[0], lat_range[1]], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND, zorder=10, facecolor='lightgrey')

    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False


def _subsample(da, skip):
    """Take every `skip`-th point along each dimension of `da` (1D or 2D coords alike)."""
    return da.isel(**{dim: slice(None, None, skip) for dim in da.dims})


def _hide_panel(ax):
    """
    Remove a panel's axis (map, ticks, border, ...) for products that don't
    provide its variable, while keeping the grid cell's space intact.

    `ax.set_visible(False)` would also drop the cell from the figure's tight
    bounding box at save time, shrinking the layout -- `axis('off')` instead
    leaves an empty (but space-occupying) cell, so the grid stays identical
    across products regardless of which panels they can fill.
    """
    ax.axis('off')


def _plot_scalar_panel(ax, dataset, years_list, var_name, cmap, title, vmin, vmax, lon_range, lat_range):
    data = _load_djf_mean(dataset, var_name, years_list)
    if data is None:
        _hide_panel(ax)
        return

    _set_map_extent(ax, lon_range, lat_range)
    im = ax.pcolormesh(data['lon'], data['lat'], data, transform=ccrs.PlateCarree(),
                       cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')

    ticks = np.linspace(vmin, vmax, 5)
    plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.08, shrink=0.8,
                 ticks=ticks, extend='both')
    ax.set_title(title, fontweight='bold')


def _plot_surface_current_panel(ax, dataset, years_list, lon_range, lat_range):
    spec = SURFACE_CURRENT_PANELS.get(dataset)
    u = v = None
    if spec is not None:
        u = _load_djf_mean(dataset, spec['u'], years_list)
        v = _load_djf_mean(dataset, spec['v'], years_list)

    if u is None or v is None:
        _hide_panel(ax)
        return

    _set_map_extent(ax, lon_range, lat_range)
    speed = np.hypot(u.values, v.values)
    strm = ax.streamplot(u['lon'].values, u['lat'].values, u.values, v.values,
                         transform=ccrs.PlateCarree(), color=speed, cmap=spec['cmap'], density=1.5)
    plt.colorbar(strm.lines, ax=ax, orientation='horizontal', pad=0.08, shrink=0.8, label=spec['label'])
    ax.set_title(spec['title'], fontweight='bold')


def _plot_wind_stress_panel(ax, dataset, years_list, lon_range, lat_range):
    spec = WIND_STRESS_PANELS.get(dataset)
    u = v = None
    if spec is not None:
        u = _load_djf_mean(dataset, spec['u'], years_list)
        v = _load_djf_mean(dataset, spec['v'], years_list)

    if u is None or v is None:
        _hide_panel(ax)
        return

    _set_map_extent(ax, lon_range, lat_range)
    u_sub, v_sub = _subsample(u, spec['skip']), _subsample(v, spec['skip'])
    q = ax.quiver(u_sub['lon'], u_sub['lat'], u_sub, v_sub,
                  transform=ccrs.PlateCarree(), scale=spec['scale'])
    ax.quiverkey(q, 0.9, 0.1, spec['key_value'], spec['key_label'], labelpos='E', coordinates='figure')
    ax.set_title(spec['title'], fontweight='bold')


def plot_one_year(dataset='ORAS5', years=2010, lon_range=(-70, -10), lat_range=(40, 70)):
    """
    Plot the average DJF state of MLD, SST, SSS, surface current (streamline),
    heat flux, freshwater flux, and wind stress (vector) for `dataset`, over
    the specified years and region.

    Panels for variables that `dataset` does not provide are hidden but keep
    their place in the fixed 2x4 grid, so the layout matches across products.

    dataset:   'ORAS5' or 'GLORYS'
    years:     int or list of ints
    lon_range: tuple (min_lon, max_lon)
    lat_range: tuple (min_lat, max_lat)
    """
    years_list = [int(years)] if isinstance(years, (int, str)) else sorted(years)

    plt.rcParams.update({'font.size': 10})

    year_label = f"{years_list[0]}" if len(years_list) == 1 else f"{years_list[0]}-{years_list[-1]}"
    print(f"Loading and averaging {dataset} data for DJF {year_label}...")

    fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(24, 10),
                             subplot_kw={'projection': ccrs.PlateCarree()})
    axes = axes.flatten()
    used_indices = set()

    for idx, var_name, cmap, title, vmin, vmax in SCALAR_PANELS:
        used_indices.add(idx)
        _plot_scalar_panel(axes[idx], dataset, years_list, var_name, cmap, title, vmin, vmax, lon_range, lat_range)

    used_indices.add(SURFACE_CURRENT_PANEL_INDEX)
    _plot_surface_current_panel(axes[SURFACE_CURRENT_PANEL_INDEX], dataset, years_list, lon_range, lat_range)

    used_indices.add(WIND_STRESS_PANEL_INDEX)
    _plot_wind_stress_panel(axes[WIND_STRESS_PANEL_INDEX], dataset, years_list, lon_range, lat_range)

    # Any grid slot nobody claimed (e.g. the trailing blank in row 2) stays
    # empty but keeps its place, just like the hidden per-product panels.
    for idx, ax in enumerate(axes):
        if idx not in used_indices:
            _hide_panel(ax)

    plt.suptitle(f'{dataset} DJF Mean ({year_label}) - SPG Region', fontsize=22, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs('figures', exist_ok=True)
    suffix = f"{years_list[0]}_{years_list[-1]}" if len(years_list) > 1 else f"{years_list[0]}"
    output_path = f'figures/{dataset.lower()}_djf_avg_{suffix}_panels.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Saved {output_path}")
    plt.close()


if __name__ == "__main__":
    box_kwargs = dict(lon_range=(-70, -10), lat_range=(40, 70))
    years_to_plot = list(range(2003, 2015))

    for dataset in ['ORAS5', 'GLORYS']:
        for year in years_to_plot:
            plot_one_year(dataset=dataset, years=year, **box_kwargs)
        plot_one_year(dataset=dataset, years=years_to_plot, **box_kwargs)
