from pathlib import Path
import traceback
import logging
from multiprocessing import Pool

NUM_WORKERS = 2  # adjust based on server rate limits / API concurrency policy

OUTPUT_ROOT = Path("data/GLORYS")

YEARS = range(2003, 2022)  # GLORYS multi-year product availability

DATASET_ID = "cmems_mod_glo_phy_my_0.083deg_P1M-m"

VARIABLES = ["uo", "vo", "thetao", "so"]

# Data extraction region (matches CLAUDE.md: 40-70N, 60-10W)
MIN_LONGITUDE = -60
MAX_LONGITUDE = -10
MIN_LATITUDE = 40
MAX_LATITUDE = 70
MIN_DEPTH = 0
MAX_DEPTH = 10000


def download_task(args):
    """Worker: download one (variable_name, year) pair."""
    variable_name, year = args

    # Imported per-worker to avoid multiprocessing pickling issues.
    import copernicusmarine

    output_dir = OUTPUT_ROOT / variable_name
    output_dir.mkdir(parents=True, exist_ok=True)

    output_filename = f"{variable_name}_{year:04d}.nc"
    output_file = output_dir / output_filename

    if output_file.exists():
        print(f"Skipping (already exists): {variable_name} - {year:04d}")
        return

    print(f"Downloading: {variable_name} - {year:04d}")
    try:
        copernicusmarine.subset(
            dataset_id=DATASET_ID,
            variables=[variable_name],
            minimum_longitude=MIN_LONGITUDE,
            maximum_longitude=MAX_LONGITUDE,
            minimum_latitude=MIN_LATITUDE,
            maximum_latitude=MAX_LATITUDE,
            start_datetime=f"{year:04d}-01-01T00:00:00",
            end_datetime=f"{year:04d}-12-31T23:59:59",
            minimum_depth=MIN_DEPTH,
            maximum_depth=MAX_DEPTH,
            output_filename=output_filename,
            output_directory=str(output_dir),
        )
    except Exception:
        print(f"Exception when downloading {variable_name} {year:04d}.")
        print(traceback.format_exc())


if __name__ == "__main__":
    logging.basicConfig(level="INFO")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    tasks = [
        (variable_name, year)
        for variable_name in VARIABLES
        for year in YEARS
    ]

    print(f"Total tasks: {len(tasks)}, workers: {NUM_WORKERS}")

    with Pool(processes=NUM_WORKERS) as pool:
        pool.map(download_task, tasks)
