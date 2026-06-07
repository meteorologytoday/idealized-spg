import os

import matplotlib as mplt
mplt.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

import data_loader as dl

# (axis label, standardized surface-variable name -- see data_loader -- or
# 'wind_stress', combined below from 'taux'/'tauy' since neither product
# stores wind-stress magnitude directly)
VARIABLES = [
    (r'SST ($^\circ$C)',                 'sst'),
    ('SSS (psu)',                        'sss'),
    ('MLD (m)',                          'mld'),
    (r'Heat Flux (W/m$^2$)',             'heat_flux'),
    (r'Freshwater Flux (mm/day)',        'water_flux'),
    (r'Wind Stress (N/m$^2$)',           'wind_stress'),
]

# Per-dataset styling for the overlay. Unlike plot_vertical_profile's
# season x variable grid -- where color already encodes the variable, so
# datasets are told apart by linestyle alone -- each panel here shows a
# single variable, leaving color free to encode the dataset too.
DATASET_STYLE = {
    'GLORYS': dict(color='tab:blue',   linestyle='-'),
    'ORAS5':  dict(color='tab:orange', linestyle='--'),
}


def _expand_year_range(year_range):
    """
    Convert an inclusive (start_year, end_year) range into the list of
    individual years `data_loader.load_surface` expects. None means "every
    year found on disk".
    """
    if year_range is None:
        return None

    start_year, end_year = year_range
    return list(range(start_year, end_year + 1))


def _load_surface_timeseries(dataset, kind, years, lat_range, lon_range):
    """
    Load one surface variable for `dataset`, average it over the given
    lat-lon box, and return the resulting 1D (time,) series -- or None if
    `dataset` doesn't provide it (e.g. GLORYS has no flux or wind-stress
    products, see data_loader._SURFACE_VARIABLES).

    kind: a standardized surface-variable name from data_loader ('sst',
          'sss', 'mld', 'heat_flux', 'water_flux'), or 'wind_stress' --
          combined from 'taux'/'tauy' as sqrt(taux^2 + tauy^2)
    """
    if kind == 'wind_stress':
        taux = dl.load_surface(dataset, 'taux', years=years)
        tauy = dl.load_surface(dataset, 'tauy', years=years)
        if taux is None or tauy is None:
            return None
        # Binary arithmetic between separately-loaded DataArrays drops their
        # 'lat'/'lon' coords (xarray treats same-valued-but-distinct coordinate
        # variables as conflicting), so box_mean would otherwise have nothing
        # to select on -- reattach them from taux afterwards.
        da = np.sqrt(taux**2 + tauy**2).assign_coords(lat=taux['lat'], lon=taux['lon'])
    else:
        da = dl.load_surface(dataset, kind, years=years)
        if da is None:
            return None

    ts = dl.box_mean(da, lat_range, lon_range).compute()
    if kind == 'water_flux':
        # kg m^-2 s^-1 -> mm/day
        ts = ts * 86400
    return ts


def _legend_handles(datasets):
    """Style-only legend handles distinguishing datasets by color/linestyle."""
    return [
        plt.Line2D([], [], color=DATASET_STYLE[dataset]['color'], linestyle=DATASET_STYLE[dataset]['linestyle'],
                   linewidth=2, label=dataset)
        for dataset in datasets
    ]


def format_coord(val, coord_type='lat'):
    if coord_type == 'lat':
        suffix = 'N' if val >= 0 else 'S'
    else:
        suffix = 'E' if val >= 0 else 'W'
    return f"{abs(val):g}{suffix}"


def plot_timeseries_all(datasets=('GLORYS', 'ORAS5'), years=None, lat_range=(50, 65), lon_range=(-60, -10)):
    """
    Plot timeseries of SST, SSS, MLD, heat flux, freshwater flux, and wind
    stress, averaged over a lat-lon box, overlaying each dataset on the same
    axes (one panel per variable) so the products can be compared directly.

    Datasets that don't provide a variable (e.g. GLORYS has no flux or
    wind-stress products) are simply skipped for that panel.

    datasets:  iterable of dataset names, each one of 'GLORYS' or 'ORAS5'
    years:     optional (start_year, end_year) inclusive range restricting
               which years to load, or None for every year found
    lat_range: (min_lat, max_lat) of the averaging box, in degrees North
    lon_range: (min_lon, max_lon) of the averaging box, in degrees East
    """
    plt.rcParams.update({'font.size': 14})

    unknown = [d for d in datasets if d not in DATASET_STYLE]
    if unknown:
        raise ValueError(f"No style defined for dataset(s) {unknown}; expected one of {sorted(DATASET_STYLE)}")

    expanded_years = _expand_year_range(years)

    fig, axes = plt.subplots(len(VARIABLES), 1, figsize=(12, 14), sharex=True)
    lat_str = f"{format_coord(lat_range[0], 'lat')}-{format_coord(lat_range[1], 'lat')}"
    lon_str = f"{format_coord(lon_range[0], 'lon')}-{format_coord(lon_range[1], 'lon')}"

    start_year, end_year = None, None

    for i, (label, kind) in enumerate(VARIABLES):
        print(f"Processing {label}...")
        plotted = False
        for dataset in datasets:
            ts = _load_surface_timeseries(dataset, kind, expanded_years, lat_range, lon_range)
            if ts is None:
                continue

            ts_start = int(ts.time.dt.year.min().values)
            ts_end = int(ts.time.dt.year.max().values)
            start_year = ts_start if start_year is None else min(start_year, ts_start)
            end_year = ts_end if end_year is None else max(end_year, ts_end)

            style = DATASET_STYLE[dataset]
            axes[i].plot(ts['time'].values, ts.values, color=style['color'], linestyle=style['linestyle'], linewidth=1.5)
            plotted = True

        if plotted:
            axes[i].set_ylabel(label, fontsize=12)
            axes[i].grid(True, linestyle='--', alpha=0.7)

        if i == 0:
            axes[i].set_title(f'Averaged Timeseries ({lat_str}, {lon_str})', fontsize=14, fontweight='bold')
            axes[i].legend(handles=_legend_handles(datasets), loc='upper right', fontsize=11)

    axes[-1].set_xlabel('Year', fontsize=12)
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(axes[-1].get_xticklabels(), rotation=45)
    plt.tight_layout()

    os.makedirs('figures', exist_ok=True)
    suffix = '_'.join(dataset.lower() for dataset in datasets)
    output_path = f'figures/{suffix}_spg_timeseries_all_{lat_str}_{lon_str}_{start_year}_{end_year}.png'
    plt.savefig(output_path, dpi=150)
    print(f"Saved {output_path}")
    plt.close()


if __name__ == "__main__":
    plot_timeseries_all(datasets=('GLORYS', 'ORAS5'), years=(2003, 2012),
                        lat_range=(60 - 2.5, 60 + 2.5), lon_range=(-55 - 2.5, -55 + 2.5))
