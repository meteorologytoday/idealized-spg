import os

import gsw
import matplotlib as mplt
mplt.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import data_loader as dl

G = 9.81        # gravitational acceleration, m s^-2
RHO0 = 1025.0   # reference density used to define buoyancy, kg m^-3

SEASONS = ['DJF', 'MAM', 'JJA', 'SON']

# (standardized variable name, axis label, line/shading color)
VAR_CONFIG = [
    ('temperature', r'Temperature ($^\circ$C)',          'tab:red'),
    ('salinity',    'Salinity (PSU)',                    'tab:blue'),
    ('buoyancy',    r'Buoyancy (m s$^{-2}$)',            'tab:green'),
    ('u_velocity',  r'Zonal velocity (m s$^{-1}$)',      'tab:purple'),
    ('v_velocity',  r'Meridional velocity (m s$^{-1}$)', 'tab:orange'),
]

# Standardized variable names loaded directly from disk; buoyancy is derived
# from temperature and salinity afterwards (see compute_buoyancy) rather than
# loaded, so it is listed in VAR_CONFIG but not here.
_LOADED_VARS = ['temperature', 'salinity', 'u_velocity', 'v_velocity']


def compute_buoyancy(theta, salinity, depth, lat_ref, lon_ref):
    """
    Buoyancy b = -g * (rho - rho0) / rho0, derived from potential temperature
    and practical salinity profiles using the TEOS-10 equation of state (gsw).

    `rho` is the *potential* density referenced to the surface (0 dbar), not
    the in-situ density: in-situ density is dominated by the compressibility
    (pressure) effect, which would produce a strong apparent stratification
    that has nothing to do with the water's thermohaline structure and would
    mask the actual buoyancy signal we want to see.

    theta, salinity: xr.DataArray with dims (time, depth)
    depth:           1D array of depths (m, positive downward)
    lat_ref, lon_ref: scalar reference position (box mean), needed to convert
                      depth to pressure and practical to absolute salinity
    """
    pressure = gsw.p_from_z(-depth, lat_ref)
    SA = gsw.SA_from_SP(salinity.values, pressure, lon_ref, lat_ref)
    CT = gsw.CT_from_pt(SA, theta.values)
    rho = gsw.rho(SA, CT, 0.0)

    buoyancy = -G * (rho - RHO0) / RHO0
    return xr.DataArray(buoyancy, dims=theta.dims, coords=theta.coords, name='buoyancy')


def _expand_year_range(year_range):
    """
    Convert an inclusive (start_year, end_year) range into the list of
    individual years `data_loader.load` expects. None means "every year
    found on disk".
    """
    if year_range is None:
        return None

    start_year, end_year = year_range
    return list(range(start_year, end_year + 1))


def _load_seasonal_profiles(dataset, year_range, lat_range, lon_range, max_depth):
    """
    Load temperature, salinity, and zonal/meridional velocity for `dataset`,
    derive buoyancy from temperature and salinity, average everything over
    the lat-lon box, and group the resulting profiles by season.

    Note that velocity components are loaded and box-averaged on their own
    native (staggered U/V) grids, exactly like temperature and salinity on
    the T grid -- consistent with this project's no-interpolation policy.

    year_range: (start_year, end_year) inclusive range, or None for every
                year found on disk -- see _expand_year_range

    Returns (seasonal_mean, seasonal_std, start_year, end_year): the seasonal_*
    values are dicts mapping standardized variable name -> xr.DataArray with
    dims (season, depth), trimmed to depths <= max_depth.
    """
    years = _expand_year_range(year_range)

    print(f"Loading {dataset} {', '.join(_LOADED_VARS)}...")
    loaded = {name: dl.load(dataset, name, years=years) for name in _LOADED_VARS}

    print(f"Averaging {dataset} over the box: lat {lat_range}, lon {lon_range} ...")
    depth_slice = slice(0, max_depth)
    profiles = {name: dl.box_mean(da, lat_range, lon_range).sel(depth=depth_slice)
                for name, da in loaded.items()}

    depth = profiles['temperature']['depth']
    lat_ref = float(np.mean(lat_range))
    lon_ref = float(np.mean(lon_range))

    print(f"Deriving {dataset} buoyancy via gsw (TEOS-10 equation of state)...")
    profiles['buoyancy'] = compute_buoyancy(profiles['temperature'], profiles['salinity'], depth.values, lat_ref, lon_ref)

    print(f"Calculating {dataset} seasonal statistics...")
    seasonal_mean = {name: da.groupby('time.season').mean(dim='time').compute() for name, da in profiles.items()}
    seasonal_std = {name: da.groupby('time.season').std(dim='time').compute() for name, da in profiles.items()}

    start_year = int(loaded['temperature'].time.dt.year.min().values)
    end_year = int(loaded['temperature'].time.dt.year.max().values)

    return seasonal_mean, seasonal_std, start_year, end_year


def _new_profile_grid(suptitle):
    """
    A season x variable grid of empty depth-profile axes, with row/column
    labels and the given suptitle already in place.

    Rows (seasons) share their depth (y) axis, inverted so depth 0 is at the
    top; columns (variables) share their data (x) axis, so that e.g. all four
    seasons' temperature panels span the same range and can be compared
    directly.
    """
    fig, axes = plt.subplots(len(SEASONS), len(VAR_CONFIG), figsize=(16 / 3 * len(VAR_CONFIG), 20),
                             sharey='row', sharex='col')
    fig.suptitle(suptitle, fontsize=20, fontweight='bold', y=0.97)

    for row, season in enumerate(SEASONS):
        for col, (_, label, _) in enumerate(VAR_CONFIG):
            ax = axes[row, col]
            ax.grid(True, alpha=0.3, linestyle='--')

            if row == 0:
                ax.set_title(label, fontsize=16, fontweight='bold')
            if col == 0:
                # Sharing the y axis means inverting it once here also flips
                # the other columns in this row -- inverting it again per
                # column would just toggle it back and forth.
                ax.invert_yaxis()
                ax.set_ylabel(f'{season}\nDepth (m)', fontsize=15, fontweight='bold')
            if row == len(SEASONS) - 1:
                ax.set_xlabel(label)

    return fig, axes


def _box_label(lat_range, lon_range):
    return (f'Box mean: lat {lat_range[0]}°-{lat_range[1]}°N, '
            f'lon {lon_range[0]}°-{lon_range[1]}°E')


def plot_vertical_profile(dataset='GLORYS', years=None, lat_range=(50, 65), lon_range=(-60, -10), max_depth=2000):
    """
    Plot seasonal vertical profiles (mean, shaded +-1 std) of temperature,
    salinity, and buoyancy, averaged over a lat-lon box, for one product.

    dataset:   'GLORYS' or 'ORAS5'
    years:     optional (start_year, end_year) inclusive range restricting
               which years to load (e.g. so that GLORYS and ORAS5 can be
               compared over a common period), or None for every year found
    lat_range: (min_lat, max_lat) of the averaging box, in degrees North
    lon_range: (min_lon, max_lon) of the averaging box, in degrees East
    max_depth: deepest level shown on the depth axis, in metres
    """
    plt.rcParams.update({'font.size': 14})

    seasonal_mean, seasonal_std, start_year, end_year = _load_seasonal_profiles(
        dataset, years, lat_range, lon_range, max_depth)

    fig, axes = _new_profile_grid(
        f'{dataset} Vertical Structure ({start_year}-{end_year})\n{_box_label(lat_range, lon_range)}')

    for row, season in enumerate(SEASONS):
        for col, (var_name, _, color) in enumerate(VAR_CONFIG):
            ax = axes[row, col]

            d = seasonal_mean[var_name]['depth'].values
            mean = seasonal_mean[var_name].sel(season=season).values
            std = seasonal_std[var_name].sel(season=season).values

            # Shading: +-1 standard deviation around the seasonal mean profile
            ax.fill_betweenx(d, mean - std, mean + std, color=color, alpha=0.25)
            # Line: seasonal mean profile
            ax.plot(mean, d, color=color, linewidth=2)

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    os.makedirs('figures', exist_ok=True)
    output_path = f'figures/{dataset.lower()}_spg_vertical_profile_{start_year}_{end_year}.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Saved {output_path}")
    plt.close()


# Per-dataset line styling for the side-by-side comparison plot. Variables
# already have their own color in VAR_CONFIG, so datasets are told apart by
# line style instead.
DATASET_STYLE = {
    'GLORYS': dict(linestyle='-'),
    'ORAS5':  dict(linestyle='--'),
}


def plot_comparison(datasets=('GLORYS', 'ORAS5'), years=None,
                    lat_range=(50, 65), lon_range=(-60, -10), max_depth=2000):
    """
    Overlay seasonal vertical profiles (mean, shaded +-1 std) of temperature,
    salinity, and buoyancy from several products in the same season x variable
    grid, so they can be compared side by side panel-for-panel.

    datasets: iterable of dataset names, each one of 'GLORYS' or 'ORAS5'
    years, lat_range, lon_range, max_depth: see plot_vertical_profile
    """
    plt.rcParams.update({'font.size': 14})

    unknown = [d for d in datasets if d not in DATASET_STYLE]
    if unknown:
        raise ValueError(f"No line style defined for dataset(s) {unknown}; expected one of {sorted(DATASET_STYLE)}")

    profiles_by_dataset = {
        dataset: _load_seasonal_profiles(dataset, years, lat_range, lon_range, max_depth)
        for dataset in datasets
    }

    period_label = ', '.join(
        f'{dataset} {start_year}-{end_year}'
        for dataset, (_, _, start_year, end_year) in profiles_by_dataset.items()
    )

    fig, axes = _new_profile_grid(
        f'Vertical Structure Comparison ({period_label})\n{_box_label(lat_range, lon_range)}')

    for row, season in enumerate(SEASONS):
        for col, (var_name, _, color) in enumerate(VAR_CONFIG):
            ax = axes[row, col]

            for dataset in datasets:
                seasonal_mean, seasonal_std, _, _ = profiles_by_dataset[dataset]
                linestyle = DATASET_STYLE[dataset]['linestyle']

                d = seasonal_mean[var_name]['depth'].values
                mean = seasonal_mean[var_name].sel(season=season).values
                std = seasonal_std[var_name].sel(season=season).values

                ax.fill_betweenx(d, mean - std, mean + std, color=color, alpha=0.15)
                ax.plot(mean, d, color=color, linestyle=linestyle, linewidth=2, label=dataset)

            if row == 0 and col == len(VAR_CONFIG) - 1:
                # Datasets are distinguished by line style only (color encodes
                # the variable), so build a style-only legend once.
                handles = [
                    plt.Line2D([], [], color='0.3', linestyle=DATASET_STYLE[dataset]['linestyle'],
                               linewidth=2, label=dataset)
                    for dataset in datasets
                ]
                ax.legend(handles=handles, loc='lower right', fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    start_year = min(start for _, _, start, _ in profiles_by_dataset.values())
    end_year = max(end for _, _, _, end in profiles_by_dataset.values())

    os.makedirs('figures', exist_ok=True)
    suffix = '_'.join(dataset.lower() for dataset in datasets)
    output_path = f'figures/{suffix}_spg_vertical_profile_comparison_{start_year}_{end_year}.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Saved {output_path}")
    plt.close()


if __name__ == "__main__":
    # Common period covered by both products, so the figures are comparable.
    box_kwargs = dict(years=(2003, 2012), lat_range=(50, 65), lon_range=(-60, -10), max_depth=2000)

    for dataset in ['GLORYS', 'ORAS5']:
        plot_vertical_profile(dataset=dataset, **box_kwargs)

    plot_comparison(datasets=('GLORYS', 'ORAS5'), **box_kwargs)
