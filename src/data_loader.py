"""
Unified loader for the ORAS5 and GLORYS ocean reanalysis products.

The two products live on different horizontal grids:
  - ORAS5  uses a curvilinear NEMO grid, with 2D coordinates `nav_lat(y, x)` /
           `nav_lon(y, x)` and dimensions `time_counter`, `deptht`, `y`, `x`.
  - GLORYS uses a regular lat-lon grid, with 1D coordinates `latitude` /
           `longitude` and dimensions `time`, `depth`, `latitude`, `longitude`.

This module does NOT regrid or interpolate either product onto a common grid.
It only standardizes dimension/coordinate/variable names so that downstream
code can select a lat-lon box and average over it without caring which product
it is looking at; xarray's broadcasting handles the 1D-vs-2D coordinate
difference automatically.

After `load()`, a DataArray exposes:
  - dims:  'time', 'depth', plus the native horizontal dims
           (`y`, `x` for ORAS5; `latitude`, `longitude` for GLORYS)
  - coords: 'lat', 'lon' on their native 1D or 2D grid
  - name:  the standardized variable name requested (e.g. 'temperature')
"""

import glob

import numpy as np
import xarray as xr

ORAS5_BASE_PATH = 'data/ORAS5'
GLORYS_BASE_PATH = 'data/GLORYS'

# Where to find each standardized variable, and what it is called on disk.
_VARIABLES = {
    'ORAS5': {
        'temperature': dict(var_dir='potential_temperature', var_name='votemper'),
        'salinity':    dict(var_dir='salinity',              var_name='vosaline'),
    },
    'GLORYS': {
        'temperature': dict(var_dir='thetao', var_name='thetao'),
        'salinity':    dict(var_dir='so',     var_name='so'),
    },
}

# Native -> standardized name, per dataset. Only names actually present are renamed.
_RENAME = {
    'ORAS5':  dict(time_counter='time', deptht='depth', nav_lat='lat', nav_lon='lon'),
    'GLORYS': dict(latitude='lat', longitude='lon'),
}

_BASE_PATH = {'ORAS5': ORAS5_BASE_PATH, 'GLORYS': GLORYS_BASE_PATH}

# ORAS5 stores one file per month (e.g. *_200301_*.nc); GLORYS stores one file
# per year (e.g. thetao_2003.nc). Both can be matched with a `*<year>*` glob,
# but ORAS5 also matches version strings like 'v0.1' on a 4-digit year, so we
# anchor the year to the filename's date field instead.
_FILE_GLOB = {
    'ORAS5':  lambda var_dir, year: f'*_{year}??_*.nc',
    'GLORYS': lambda var_dir, year: f'*{year}*.nc',
}


def _find_files(dataset, var_dir, years, base_path):
    if years is None:
        return sorted(glob.glob(f'{base_path}/{var_dir}/*.nc'))

    years = [years] if isinstance(years, (int, str)) else years
    files = set()
    for year in years:
        pattern = _FILE_GLOB[dataset](var_dir, year)
        files.update(glob.glob(f'{base_path}/{var_dir}/{pattern}'))
    return sorted(files)


def load(dataset, variable, years=None, base_path=None):
    """
    Load a standardized 3D variable from one of the reanalysis products.

    dataset:   'ORAS5' or 'GLORYS'
    variable:  'temperature' or 'salinity'
    years:     optional int or iterable of ints restricting which files to load
    base_path: override the default base path for the dataset

    Returns an xr.DataArray on the dataset's native horizontal grid, with
    dimensions/coordinates renamed to the standardized 'time', 'depth', 'lat',
    'lon' and the requested standardized name.
    """
    if dataset not in _VARIABLES:
        raise ValueError(f"Unknown dataset '{dataset}', expected one of {sorted(_VARIABLES)}")
    if variable not in _VARIABLES[dataset]:
        raise ValueError(
            f"Unknown variable '{variable}' for {dataset}, "
            f"expected one of {sorted(_VARIABLES[dataset])}"
        )

    spec = _VARIABLES[dataset][variable]
    base_path = base_path or _BASE_PATH[dataset]

    files = _find_files(dataset, spec['var_dir'], years, base_path)
    if not files:
        raise FileNotFoundError(f"No files found for {dataset}/{variable} in {base_path}/{spec['var_dir']}/")

    concat_dim = 'time_counter' if dataset == 'ORAS5' else 'time'
    ds = xr.open_mfdataset(files, combine='nested', concat_dim=concat_dim)

    rename = {native: std for native, std in _RENAME[dataset].items() if native in ds.variables}
    ds = ds.rename(rename)

    return ds[spec['var_name']].rename(variable)


def box_mean(da, lat_range, lon_range):
    """
    Area-weighted (cosine-latitude) horizontal mean of `da` over a lat-lon box,
    computed on the data's native grid -- no regridding/interpolation involved.

    Works whether 'lat'/'lon' are 1D (regular grid, e.g. GLORYS) or 2D
    (curvilinear grid, e.g. ORAS5): xarray broadcasts the mask and weights
    against `da`'s horizontal dimensions either way, and the weighted mean
    collapses whichever horizontal dims remain (anything that isn't 'time' or
    'depth').
    """
    horizontal_dims = [d for d in da.dims if d not in ('time', 'depth')]

    def _in_box(lat, lon):
        return (
            (lat >= lat_range[0]) & (lat <= lat_range[1]) &
            (lon >= lon_range[0]) & (lon <= lon_range[1])
        )

    # Trim to the index bounding box of the selection first. The SPG box is a
    # small corner of either product's global grid (especially ORAS5's
    # 1021x1442 curvilinear grid), so cropping before the weighted
    # multiply-and-sum below keeps that reduction from being carried out over
    # the full horizontal grid.
    in_box = _in_box(da['lat'], da['lon'])
    bbox = {}
    for dim in horizontal_dims:
        other_dims = [d for d in in_box.dims if d != dim]
        present_along_dim = in_box.any(dim=other_dims) if other_dims else in_box
        idx = np.flatnonzero(present_along_dim.values)
        bbox[dim] = slice(idx.min(), idx.max() + 1)
    da = da.isel(**bbox)

    lat, lon = da['lat'], da['lon']
    weights = np.cos(np.deg2rad(lat)) * _in_box(lat, lon)

    return da.weighted(weights).mean(dim=horizontal_dims)
