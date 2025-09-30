import os    

from argparse import ArgumentParser
import yaml
import logging

import importlib

from pathlib import Path
from copy import deepcopy

from huggingface_hub import hf_hub_download
from osgeo import gdal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.raiseExceptions = False

import os
os.environ['GTIFF_SRS_SOURCE'] = 'EPSG'  # Prefer official EPSG definitions

gdal.UseExceptions()
gdal.SetConfigOption("GDAL_PAM_ENABLED", "NO")

def delineate(args, verbose):
    # loading model
    models_dict = {
        "small": {
            "repo_id": "MykolaL/DelineateAnything",
            "filename": "DelineateAnything-S.pt"
        },
        "large": {
            "repo_id": "MykolaL/DelineateAnything",
            "filename": "DelineateAnything.pt"
        }
    }

    model_names =  args["config"]["model"]
    if isinstance(model_names, str):
        model_name = model_names
        if model_name in models_dict:
            model_path = hf_hub_download(repo_id=models_dict[model_name]["repo_id"], filename=models_dict[model_name]["filename"])
        else:
            if os.path.exists(model_name):
                model_path = model_name
            else:
                raise ValueError(f"Unknown model '{model_name}' in configuration file.")
    else:
        model_path = []
        for model_name in model_names:
            if model_name in models_dict:
                model_path.append(hf_hub_download(repo_id=models_dict[model_name]["repo_id"], filename=models_dict[model_name]["filename"]))
            else:
                if os.path.exists(model_name):
                    model_path.append(model_name)
                else:
                    raise ValueError(f"Unknown model '{model_name}' in configuration file.")

    # loading delineation method
    method = args["config"]["method"]
    module = importlib.import_module(f"methods.{method}.inference")

    args["config"]["execution_args"] = {
        "src_folder": args["input"],
        "temp_folder": args["temp"],
        "output_path": args["output"],
        "keep_temp": args["keep_temp"],
        "mask_filepath": args["mask"]
    }

    module.execute(model_path, args["config"], verbose)

def deep_override(a, b):
    """Recursively write dict b into dict a and return the result."""
    if not isinstance(b, dict):
        return b

    result = dict(a)
    for key, b_value in b.items():
        if key in result:
            a_value = result[key]
            if isinstance(a_value, dict) and isinstance(b_value, dict):
                result[key] = deep_override(a_value, b_value)
            else:
                result[key] = b_value 
        else:
            result[key] = b_value

    return result

def batch_routine(args):
    """Parse/validate entered arguments and perform delineation in case of batch execution."""

    if not os.path.exists(args.batch_config):
        raise ValueError(f"{args.batch_config} is not exists.")
    
    batch_config = yaml.safe_load(Path(args.batch_config).read_text())
    base_config = yaml.safe_load(Path(batch_config["base_config"]).read_text())

    overrides = {}
    if "override" in batch_config and batch_config["override"]:
        for obj in batch_config["override"]:
            overrides[obj["entry"]] = obj

    setups = []
    for folder in os.listdir(batch_config["data_root"]):
        if not os.path.isdir(os.path.join(batch_config["data_root"], folder)):
            continue

        if "include" in batch_config and batch_config["include"] and not folder in batch_config["include"]:
            logger.warning(f"{folder} is not present in include list. Skipped.")
            continue

        if "exclude" in batch_config and batch_config["exclude"] and folder in batch_config["exclude"]:
            logger.warning(f"{folder} is present in exclude list. Skipped.")
            continue

        config = deepcopy(base_config)
        mask_path = os.path.join(batch_config["mask_root"], folder + ".tif")

        if folder in overrides:
            override = overrides[folder]

            if "mask" in override:
                mask_path = override["mask"]

            # load overrided config file
            if "config" in override:
                config = yaml.safe_load(Path(override["config"].read_text()))

            # make changes to currently used config file
            if "config_override" in override and override["config_override"]:
                config = deep_override(config, override["config_override"])

        if not os.path.exists(mask_path):
            logger.warning(f"Mask at {mask_path} is not existent. Mask skipped.")
            mask_path = None

        setups.append({
            "name": folder,
            "config": config,
            "input": os.path.join(batch_config["data_root"], folder),
            "output": os.path.join(batch_config["output_root"], folder + ".gpkg"),
            "temp": batch_config["temp_root"],
            "keep_temp": batch_config["keep_temp"],
            "mask": mask_path
        })
        logger.info(f"{folder} is queued for delineation.")

    for setup in setups:
        delineate(setup, args.verbose)

def main():
    parser = ArgumentParser()

    parser.add_argument("-b", "--batch", dest="batch_config", default=None,
                help="Configuration of batch execution.")
    
    parser.add_argument("--keep-temp", dest="keep_temp", action="store_true",
                    help="To save temporary files.")
    
    parser.add_argument("--verbose", dest="verbose", action="store_true",
                    help="To display trace level logs. Mostly for dev purposes.")

    args = parser.parse_args()

    if args.batch_config:
        batch_routine(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()