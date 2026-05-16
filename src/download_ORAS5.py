from pathlib import Path
import traceback
import logging
import zipfile
import os
from multiprocessing import Pool

NUM_WORKERS = 2  # adjust based on server rate limits / API concurrency policy

OUTPUT_ROOT = Path("data/ORAS5")

YEARS = range(2003, 2013)

VARIABLE_SPEC = {
    "single_level": [
        "meridional_wind_stress",
        "mixed_layer_depth_0_01",
        "mixed_layer_depth_0_03",
        "net_downward_heat_flux",
        "net_upward_water_flux",
        "sea_ice_concentration",
        "sea_ice_thickness",
        "sea_surface_height",
        "sea_surface_salinity",
        "sea_surface_temperature",
        "zonal_wind_stress",
    ],
    "all_levels": [
        "meridional_velocity",
        "potential_temperature",
        "salinity",
        "zonal_velocity",
    ],
}


def all_months_exist(output_dir, year):
    """Return True if all 12 monthly NetCDF files for the given year are present."""
    return all(
        len(list(output_dir.glob(f"*_{year:04d}{month:02d}_*.nc"))) > 0
        for month in range(1, 13)
    )


def unzip_and_remove(zip_file_path, extract_to_dir):
    try:
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(extract_to_dir)
            print(f"Successfully extracted to: {extract_to_dir}")
        os.remove(zip_file_path)
        print(f"Successfully deleted: {zip_file_path}")
    except FileNotFoundError:
        print("The specified zip file was not found.")
    except zipfile.BadZipFile:
        print("The file is not a valid ZIP archive.")
    except Exception as e:
        print(f"An error occurred: {e}")


def download_task(args):
    """Worker: download one (vertical_resolution, variable_name, year) triplet."""
    vertical_resolution, variable_name, year = args

    tmp_dir = OUTPUT_ROOT / "_tmp_" / variable_name
    tmp_dir.mkdir(parents=True, exist_ok=True)

    output_dir = OUTPUT_ROOT / variable_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if all_months_exist(output_dir, year):
        print(f"Skipping (already complete): {variable_name} - {year:04d}")
        return

    # Client is created per-worker to avoid pickling issues.
    from ecmwf.datastores import Client
    client = Client()

    tmp_file = tmp_dir / f"{variable_name}_year-{year:04d}.zip"
    print(f"Downloading: {variable_name} - {year:04d}")

    try:
        request = {
            "product_type": ["consolidated"],
            "vertical_resolution": vertical_resolution,
            "variable": variable_name,
            "year": [f"{year:04d}"],
            "month": [
                "01", "02", "03",
                "04", "05", "06",
                "07", "08", "09",
                "10", "11", "12",
            ],
            "data_format": "netcdf",
            "download_format": "unarchived",
        }
        client.retrieve("reanalysis-oras5", request, target=str(tmp_file))
        print(f"Unzip {tmp_file} => {output_dir}")
        unzip_and_remove(str(tmp_file), str(output_dir))

    except Exception:
        print(f"Exception when downloading {variable_name} {year:04d}.")
        print(traceback.format_exc())


if __name__ == "__main__":
    logging.basicConfig(level="INFO")

    from ecmwf.datastores import Client
    Client().check_authentication()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    tasks = [
        (vertical_resolution, variable_name, year)
        for vertical_resolution, variable_names in VARIABLE_SPEC.items()
        for variable_name in variable_names
        for year in YEARS
    ]

    print(f"Total tasks: {len(tasks)}, workers: {NUM_WORKERS}")

    with Pool(processes=NUM_WORKERS) as pool:
        pool.map(download_task, tasks)
