import os

import matplotlib as mplt
mplt.use("Agg")
import matplotlib.pyplot as plt

import data_loader as dl
import cross_section as cs

SEASONS = ['DJF', 'MAM', 'JJA', 'SON']
ANNUAL = 'ANNUAL'

# (standardized variable name, colormap, axis label, vmin, vmax)
VAR_CONFIG = [
    ('temperature', 'RdYlBu_r', r'Temperature ($^\circ$C)',          0,    15),
    ('salinity',    'YlGnBu',   'Salinity (PSU)',                    34,   36),
    ('u_velocity',  'RdBu_r',   r'Zonal velocity (m s$^{-1}$)',      -0.3, 0.3),
    ('v_velocity',  'RdBu_r',   r'Meridional velocity (m s$^{-1}$)', -0.3, 0.3),
]

_LOADED_VARS = [var_name for var_name, *_ in VAR_CONFIG]


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


def _season_label(season):
    return 'Annual' if season == ANNUAL else season


def _load_cross_sections(dataset, year_range, season, point_a, point_b, max_depth, half_width):
    """
    Load temperature, salinity, zonal/meridional velocity, and mixed layer
    depth for `dataset`, project everything onto the section from `point_a`
    to `point_b`, optionally restrict to one season's months, and average
    over time.

    Note that velocity components are loaded and projected on their own
    native (staggered U/V) grids, exactly like temperature and salinity on
    the T grid -- consistent with this project's no-interpolation policy.

    year_range: (start_year, end_year) inclusive range, or None for every
                year found on disk -- see _expand_year_range
    season:     one of 'DJF'/'MAM'/'JJA'/'SON' to restrict to that season's
                months, or ANNUAL ('ANNUAL') to average over all months

    Returns (mean, mld_mean, start_year, end_year): `mean` maps standardized
    variable name -> xr.DataArray with dims (depth, distance), trimmed to
    depths <= max_depth; `mld_mean` is an xr.DataArray with dims (distance,).
    """
    years = _expand_year_range(year_range)

    print(f"Loading {dataset} {', '.join(_LOADED_VARS)}, mld...")
    loaded = {name: dl.load(dataset, name, years=years) for name in _LOADED_VARS}
    mld = dl.load_surface(dataset, 'mld', years=years)

    print(f"Projecting {dataset} onto the cross-section "
          f"({point_a[0]}°N, {point_a[1]}°E) - ({point_b[0]}°N, {point_b[1]}°E) ...")
    depth_slice = slice(0, max_depth)
    sections = {name: cs.project_cross_section(da, point_a, point_b, half_width).sel(depth=depth_slice)
                for name, da in loaded.items()}
    mld_section = cs.project_cross_section(mld, point_a, point_b, half_width)

    start_year = int(loaded['temperature'].time.dt.year.min().values)
    end_year = int(loaded['temperature'].time.dt.year.max().values)

    if season != ANNUAL:
        sections = {name: da.sel(time=da.time.dt.season == season) for name, da in sections.items()}
        mld_section = mld_section.sel(time=mld_section.time.dt.season == season)

    print(f"Averaging {dataset} over {_season_label(season)} ...")
    mean = {name: da.mean(dim='time').compute() for name, da in sections.items()}
    mld_mean = mld_section.mean(dim='time').compute()

    return mean, mld_mean, start_year, end_year


def _new_sections_grid(n_rows, suptitle):
    """
    A grid of empty depth-vs-distance axes, one row per cross-section and
    one column per variable, with column titles on the top row and the
    given suptitle already in place.

    Every panel shares the depth (y) and distance (x) axes -- inverted so
    depth 0 is at the top -- so variables and cross-sections can all be
    compared directly on the same scales.

    The fixed amount of space the suptitle and column titles need is a
    shrinking fraction of the figure as rows are added (the figure grows
    taller but that header doesn't), hence the 1/n_rows scaling -- chosen
    so a single row reproduces the original layout (y=1.08).
    """
    n_cols = len(VAR_CONFIG)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16 / 3 * n_cols, 5 * n_rows),
                             sharey=True, sharex=True, squeeze=False)
    fig.suptitle(suptitle, fontsize=20, fontweight='bold', y=1 + 0.08 / n_rows)

    for row in range(n_rows):
        for col, (_, _, label, _, _) in enumerate(VAR_CONFIG):
            ax = axes[row, col]
            ax.grid(True, alpha=0.3, linestyle='--')

            if row == 0:
                ax.set_title(label, fontsize=16, fontweight='bold')
            if row == n_rows - 1:
                ax.set_xlabel('Distance along section (km)')
            if col == 0:
                if row == 0:
                    # Sharing the y axis means inverting it once here also
                    # flips every other panel -- inverting it again per
                    # panel would just toggle it back and forth.
                    ax.invert_yaxis()
                ax.set_ylabel('Depth (m)', fontsize=15, fontweight='bold')

    return fig, axes


def _new_section_grid(suptitle):
    """
    A single row of empty depth-vs-distance axes, one per variable -- see
    `_new_sections_grid`, of which this is the one-row case.
    """
    fig, axes = _new_sections_grid(1, suptitle)
    return fig, axes[0]


def _section_label(point_a, point_b):
    return (f'Cross-section: ({point_a[0]}°N, {point_a[1]}°E) '
            f'→ ({point_b[0]}°N, {point_b[1]}°E)')


def _section_tag(point_a, point_b):
    """
    Filesystem-safe identifier for the section's endpoints, e.g.
    '60N70W-60N40W' -- so cross-sections along different lines don't
    overwrite each other's output files.
    """
    def _point_tag(point):
        lat, lon = point
        return f"{lat:g}{'n' if lat >= 0 else 's'}{abs(lon):g}{'e' if lon >= 0 else 'w'}"

    return f'{_point_tag(point_a)}-{_point_tag(point_b)}'


def plot_cross_section(dataset='GLORYS', years=None, season='DJF', point_a=(60, -70), point_b=(60, -40),
                       max_depth=2000, half_width=0.25):
    """
    Plot depth-vs-distance cross-sections of temperature, salinity, and
    zonal/meridional velocity along the line from `point_a` to `point_b`,
    for one product and one season (or the annual mean), with the mixed
    layer depth overlaid as a line.

    dataset:    'GLORYS' or 'ORAS5'
    years:      optional (start_year, end_year) inclusive range restricting
                which years to load, or None for every year found
    season:     one of 'DJF', 'MAM', 'JJA', 'SON' to average over that
                season's months, or ANNUAL ('ANNUAL') to average over the
                full year
    point_a, point_b: (lat, lon) endpoints of the section, in degrees
    max_depth:  deepest level shown on the depth axis, in metres
    half_width: half-width, in degrees, of the lat-lon box averaged around
                each point along the section -- see cross_section.project_cross_section
    """
    if season != ANNUAL and season not in SEASONS:
        raise ValueError(f"Unknown season '{season}', expected one of {SEASONS} or {ANNUAL!r} for the annual mean")

    plt.rcParams.update({'font.size': 14})

    mean, mld_mean, start_year, end_year = _load_cross_sections(
        dataset, years, season, point_a, point_b, max_depth, half_width)

    season_label = _season_label(season)
    fig, axes = _new_section_grid(
        f'{dataset} {season_label} Cross-section ({start_year}-{end_year})\n{_section_label(point_a, point_b)}')

    for col, (var_name, cmap, _, vmin, vmax) in enumerate(VAR_CONFIG):
        ax = axes[col]

        data = mean[var_name].transpose('depth', 'distance')
        im = ax.pcolormesh(data['distance'], data['depth'], data,
                           cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
        plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.18, shrink=0.85, extend='both')

        ax.plot(mld_mean['distance'], mld_mean, color='k', linewidth=2, label='MLD')

        if col == len(VAR_CONFIG) - 1:
            ax.legend(loc='lower right', fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.86])

    os.makedirs('figures', exist_ok=True)
    suffix = f'{_section_tag(point_a, point_b)}_{season_label.lower()}_{start_year}_{end_year}'
    output_path = f'figures/{dataset.lower()}_spg_cross_section_{suffix}.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Saved {output_path}")
    plt.close()


def plot_cross_sections(dataset='GLORYS', years=None, season='DJF',
                        sections=(((60, -70), (60, -40)),), max_depth=2000, half_width=0.25):
    """
    Plot depth-vs-distance cross-sections of temperature, salinity, and
    zonal/meridional velocity along several transects, stacked vertically
    in a single figure -- one row per transect, top to bottom in the given
    order, sharing the depth and distance axes so transects and variables
    can all be compared directly, with the mixed layer depth overlaid as a
    line on every panel.

    dataset, years, season, max_depth, half_width: see plot_cross_section
    sections: sequence of (point_a, point_b) pairs -- (lat, lon) tuples in
              degrees, top to bottom -- defining the transects to plot;
              each pair is independent and can run in any direction (zonal,
              meridional, or an arbitrary diagonal), exactly like
              plot_cross_section's point_a/point_b
    """
    if season != ANNUAL and season not in SEASONS:
        raise ValueError(f"Unknown season '{season}', expected one of {SEASONS} or {ANNUAL!r} for the annual mean")

    plt.rcParams.update({'font.size': 14})

    rows = []
    for point_a, point_b in sections:
        mean, mld_mean, start_year, end_year = _load_cross_sections(
            dataset, years, season, point_a, point_b, max_depth, half_width)
        rows.append((point_a, point_b, mean, mld_mean))

    season_label = _season_label(season)
    fig, axes = _new_sections_grid(
        len(rows), f'{dataset} {season_label} Cross-sections ({start_year}-{end_year})')

    for row, (point_a, point_b, mean, mld_mean) in enumerate(rows):
        for col, (var_name, cmap, _, vmin, vmax) in enumerate(VAR_CONFIG):
            ax = axes[row, col]

            data = mean[var_name].transpose('depth', 'distance')
            im = ax.pcolormesh(data['distance'], data['depth'], data,
                               cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
            plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.18, shrink=0.85, extend='both')

            ax.plot(mld_mean['distance'], mld_mean, color='k', linewidth=2, label='MLD')

            if row == 0 and col == len(VAR_CONFIG) - 1:
                ax.legend(loc='lower right', fontsize=12)

        # Row label in the left margin, identifying which transect this row
        # is -- the column titles (e.g. 'Temperature') already say what's
        # plotted, so this is the piece that's otherwise missing per row.
        # Kept short (no 'Cross-section:' prefix, as the suptitle already
        # establishes that) so the rotated text fits within the row's
        # height without overlapping its neighbours.
        row_label = f'({point_a[0]}°N, {point_a[1]}°E) → ({point_b[0]}°N, {point_b[1]}°E)'
        axes[row, 0].annotate(row_label, xy=(-0.4, 0.5), xycoords='axes fraction',
                              rotation=90, va='center', ha='center', fontsize=13, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 1 - 0.14 / len(rows)])

    os.makedirs('figures', exist_ok=True)
    tag = '_'.join(_section_tag(point_a, point_b) for point_a, point_b in sections)
    suffix = f'{tag}_{season_label.lower()}_{start_year}_{end_year}'
    output_path = f'figures/{dataset.lower()}_spg_cross_sections_{suffix}.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Saved {output_path}")
    plt.close()


if __name__ == "__main__":
    # Cross-section through the Labrador Sea -- crosses the boundary current
    # on the western side of the SPG, where deep winter mixing is strongest.
    box_kwargs = dict(years=(2003, 2012), point_a=(60, -70), point_b=(60, -40), max_depth=2000)

    for dataset in ['GLORYS', 'ORAS5']:
        plot_cross_section(dataset=dataset, season='DJF', **box_kwargs)

    # A series of zonal transects at several latitudes, stacked into one
    # figure (northernmost on top) so the SPG's structure can be compared
    # across latitude bands at a glance -- plot_cross_sections takes the
    # transects as plain (point_a, point_b) pairs, so this works just as
    # well for north-south or diagonal transects, not just zonal ones.
    lat_sections = tuple(((lat, -70), (lat, -40)) for lat in [67.5, 65, 62.5, 60])
    plot_cross_sections(dataset='GLORYS', years=(2003, 2012), season='DJF',
                        sections=lat_sections, max_depth=2000)
