"""
Project gridded (lat, lon, ...) data onto a straight cross-section line
defined by two (lat, lon) endpoints.

The line is divided into a series of small lat-lon boxes laid edge-to-edge
from one endpoint to the other; the data is averaged within each box (area-
weighted by cos(latitude), exactly like data_loader.box_mean) to produce one
sample per box. This works uniformly for 1D regular grids (e.g. GLORYS) and
2D curvilinear grids (e.g. ORAS5), and -- like the rest of this project --
never regrids/interpolates the source data, it only averages over small
patches of the native grid.
"""

import numpy as np
import xarray as xr

EARTH_RADIUS_KM = 6371.0


def _haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance (km) between points given in degrees."""
    lat1, lon1, lat2, lon2 = (np.deg2rad(x) for x in (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def section_points(point_a, point_b, half_width):
    """
    Lay out sample points along the straight line (in lat-lon space) from
    `point_a` to `point_b`, spaced `2 * half_width` degrees apart so that the
    averaging boxes built around them (see `project_cross_section`) sit
    edge-to-edge without gaps or overlap.

    point_a, point_b: (lat, lon) endpoints of the section, in degrees
    half_width:       half the spacing between sample points, in degrees --
                      equivalently, half the width of each averaging box

    Returns (lat, lon, distance): 1D arrays of sample-point coordinates and
    their great-circle distance from `point_a`, in km.
    """
    lat_a, lon_a = point_a
    lat_b, lon_b = point_b

    line_length = np.hypot(lat_b - lat_a, lon_b - lon_a)
    n_points = max(int(round(line_length / (2 * half_width))) + 1, 2)

    t = np.linspace(0.0, 1.0, n_points)
    lat = lat_a + t * (lat_b - lat_a)
    lon = lon_a + t * (lon_b - lon_a)
    distance = _haversine_km(lat_a, lon_a, lat, lon)

    return lat, lon, distance


def project_cross_section(da, point_a, point_b, half_width=0.25):
    """
    Project `da` onto the straight line from `point_a` to `point_b`, replacing
    its horizontal dimensions with a single 'distance' dimension that runs
    along the section.

    The line is sampled at points `2 * half_width` degrees apart (see
    `section_points`); around each sample point (clat, clon) a box
    (clat +- half_width, clon +- half_width) is laid down and `da` is
    averaged within it -- area-weighted by cos(latitude), exactly like
    `data_loader.box_mean` -- giving one cross-section sample per box.

    da:         xr.DataArray with 'lat'/'lon' coordinates on its horizontal
                dims, either 1D (regular grid, e.g. GLORYS) or 2D
                (curvilinear grid, e.g. ORAS5)
    point_a, point_b: (lat, lon) endpoints of the section, in degrees
    half_width: half-width, in degrees, of each averaging box (applied to
                both lat and lon); also sets the spacing between sample
                points, see `section_points`

    Returns `da` with its horizontal dims collapsed into a 'distance' dim
    (great-circle km from `point_a`).
    """
    lat_a, lon_a = point_a
    lat_b, lon_b = point_b

    lat_centers, lon_centers, distance = section_points(point_a, point_b, half_width)

    # Pre-crop to the bounding box of the whole section (padded by one box
    # half-width) -- mirrors data_loader.box_mean's critical pre-crop
    # optimization, so the per-point box means below run over a small region
    # rather than ORAS5's full 1021x1442 curvilinear grid.
    horizontal_dims = [d for d in da.dims if d not in ('time', 'depth')]
    lat_min, lat_max = sorted((lat_a, lat_b))
    lon_min, lon_max = sorted((lon_a, lon_b))
    in_section_box = (
        (da['lat'] >= lat_min - half_width) & (da['lat'] <= lat_max + half_width) &
        (da['lon'] >= lon_min - half_width) & (da['lon'] <= lon_max + half_width)
    )
    bbox = {}
    for dim in horizontal_dims:
        other_dims = [d for d in in_section_box.dims if d != dim]
        present_along_dim = in_section_box.any(dim=other_dims) if other_dims else in_section_box
        idx = np.flatnonzero(present_along_dim.values)
        bbox[dim] = slice(idx.min(), idx.max() + 1)
    da = da.isel(**bbox)

    lat, lon = da['lat'], da['lon']
    lat_weights = np.cos(np.deg2rad(lat))

    profiles = []
    for clat, clon in zip(lat_centers, lon_centers):
        in_box = (
            (lat >= clat - half_width) & (lat <= clat + half_width) &
            (lon >= clon - half_width) & (lon <= clon + half_width)
        )
        profiles.append(da.weighted(lat_weights * in_box).mean(dim=horizontal_dims))

    projected = xr.concat(profiles, dim='distance')
    return projected.assign_coords(distance=('distance', distance))
