from pathlib import Path
import traceback
import logging

from ecmwf.datastores import Client

import zipfile
import os

def unzip_and_remove(zip_file_path, extract_to_dir):
    try:
        # 1. Unzip the file
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to_dir)
            print(f"Successfully extracted to: {extract_to_dir}")
        
        # 2. Remove the original zip file
        os.remove(zip_file_path)
        print(f"Successfully deleted: {zip_file_path}")
        
    except FileNotFoundError:
        print("The specified zip file was not found.")
    except zipfile.BadZipFile:
        print("The file is not a valid ZIP archive.")
    except Exception as e:
        print(f"An error occurred: {e}")


client = Client()
client.check_authentication()

logging.basicConfig(level="INFO")

output_root = Path("data/ORAS5")
output_root.mkdir(parents=True, exist_ok=True)

years = range(2003, 2026)

variable_spec = {
    "single_level" : [
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

    "all_levels" : [
        "meridional_velocity",
        "potential_temperature",
        "salinity",
        "zonal_velocity",
    ],
}


for vertical_resolution, variable_names in variable_spec.items():
    for variable_name in variable_names:

        tmp_dir = output_root / "_tmp_" / variable_name
        tmp_dir.mkdir(parents=True, exist_ok=True)
        
        output_dir = output_root / variable_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for year in years:
            
            tmp_file = tmp_dir / f"{variable_name}_year-{year:04d}.zip"
            print(f"Downloading: {variable_name:s} - {year:04d}")
            
            try:
                collection_id = "reanalysis-oras5"
                request = {
                    "product_type": ["consolidated"],
                    "vertical_resolution": vertical_resolution,
                    "variable": variable_name,
                    "year": [ f"{year:04d}", ],
                    "month": [
                        "01", "02", "03",
                        "04", "05", "06",
                        "07", "08", "09",
                        "10", "11", "12",
                    ],
                    "data_format": "netcdf",
                    'download_format': 'unarchived',
                }
                client.retrieve(collection_id, request, target=str(tmp_file))

                print(f"Unzip {str(tmp_file):s} => {str(output_dir)}")
                unzip_and_remove(str(tmp_file), str(output_dir))
            
            except Exception as e:

                print(f"Exception happens when downloading {variable_name}.") 
                print(traceback.format_exc()) 
         
