from datetime import datetime, timedelta, timezone
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import importlib.util
import os
from pydantic import Field
from fiber.logging_utils import get_logger
import importlib.util
from uuid import uuid4
from sqlalchemy import text
from basemodel import SoilModel
from custommodel import CustomSoilModel
import traceback
import base64
import json
import asyncio
import tempfile
import math
import glob
from collections import defaultdict
import torch
from huggingface_hub import hf_hub_download
import rasterio
from utils.soil_apis import get_soil_data
from utils.region_selection import select_random_region
from utils.inference_class import SoilMoistureInferencePreprocessor
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from utils.soil_scoring_mechanism import SoilScoringMechanism

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = get_logger(__name__)
preprocessor = SoilMoistureInferencePreprocessor()
scoring_mechanism = SoilScoringMechanism()

async def main():
    # Load data
    current_time = datetime.now(timezone.utc) - timedelta(days=13) ##!! BACK TESTING 10 days ago
    target_smap_time = get_smap_time(current_time)
    ifs_forecast_time = get_ifs_time_for_smap(target_smap_time)

    _h3_data = _load_h3_map()
    _base_cells = [
        {"index": cell["index"], "resolution": cell["resolution"]}
        for cell in _h3_data["base_cells"]
    ]
    _urban_cells = set(
        cell["index"] for cell in _h3_data["urban_overlay_cells"]
    )
    _lakes_cells = set(
        cell["index"] for cell in _h3_data["lakes_overlay_cells"]
    )
    soil_data = None
    while soil_data is None:
        bbox = select_random_region(
            base_cells=_base_cells,
            urban_cells_set=_urban_cells,
            lakes_cells_set=_lakes_cells,
        )
        soil_data = await get_soil_data(bbox, ifs_forecast_time)
    tiff_path, bounds, crs = soil_data
    region = {
        "datetime": target_smap_time,
        "bbox": bbox,
        "combined_data": tiff_path,
        "sentinel_bounds": bounds,
        "sentinel_crs": crs,
        "array_shape": (222, 222),
    }
    with open(region["combined_data"], "rb") as f:
        combined_data_bytes = f.read()
    encoded_data = base64.b64encode(combined_data_bytes)
    target_time = target_smap_time.isoformat()
    data = {
            "combined_data": encoded_data.decode("ascii"),
            "sentinel_bounds": region["sentinel_bounds"],
            "sentinel_crs": region["sentinel_crs"],
            "target_time": target_time,
        }
    processed_data = await process_data(data)
    logger.info(f"Processed data shape: {processed_data['sentinel_ndvi'].shape}")
    # Base model inference
    model = _load_model(device)
    predictions = model.run_inference(processed_data)
    logger.info(f"predictions surface data shape: {predictions['surface'].shape}")
    base_score = await calculate_score(data, predictions, current_time, target_smap_time, 'base')
    logger.info(f"Base score: {base_score}")

    # Custom model inference
    custom_model = CustomSoilModel()
    custom_predictions = custom_model.run_inference(processed_data)
    custom_score = await calculate_score(data, custom_predictions, current_time, target_smap_time, 'custom')
    logger.info(f"Custom score: {custom_score}")

async def calculate_score(data, predictions, current_time, target_smap_time, image_name):
    # Plot predictions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    surface_plot = ax1.imshow(predictions["surface"], cmap='viridis')
    ax1.set_title('Surface Soil Moisture')
    plt.colorbar(surface_plot, ax=ax1, label='Moisture Content')

    rootzone_plot = ax2.imshow(predictions["rootzone"], cmap='viridis')
    ax2.set_title('Root Zone Soil Moisture')
    plt.colorbar(rootzone_plot, ax=ax2, label='Moisture Content')

    plt.tight_layout()
    plt.savefig(f'{image_name}.png')
    plt.close()

    prediction_time = get_next_preparation_time(current_time)

    prediction_all_regions = {
        "surface_sm": predictions["surface"].tolist(),
        "rootzone_sm": predictions["rootzone"].tolist(),
        "uncertainty_surface": None,
        "uncertainty_rootzone": None,
        "sentinel_bounds": data["sentinel_bounds"],
        "sentinel_crs": data["sentinel_crs"],
        "target_time": prediction_time.isoformat(),
    }
    sentinel_crs = int(str(prediction_all_regions["sentinel_crs"]).split(":")[-1])
    answer = {
        "bounds": prediction_all_regions["sentinel_bounds"],
        "crs": sentinel_crs,
        "predictions": prediction_all_regions,
        "target_time": target_smap_time,
        "region": {
            "id": 0
        },
        "contributor_id": ""
    }

    return await scoring_mechanism.score(answer)

def _load_h3_map():
        """Load H3 map data, first checking locally then from HuggingFace."""
        local_path = "./data/h3_map/full_h3_map.json"

        try:
            if os.path.exists(local_path):
                with open(local_path, "r") as f:
                    return json.load(f)

            logger.info("Local H3 map not found, downloading from HuggingFace...")
            map_path = hf_hub_download(
                repo_id="Nickel5HF/gaia_h3_mapping",
                filename="full_h3_map.json",
                repo_type="dataset",
                local_dir="./data/h3_map",
            )
            with open(map_path, "r") as f:
                return json.load(f)

        except Exception as e:
            logger.error(f"Error accessing H3 map: {str(e)}")
            logger.info("Using fallback local map...")
            raise RuntimeError("No H3 map available")

def get_smap_time(current_time: datetime) -> datetime:
    """Get SMAP time based on validator execution time."""
    smap_hours = [1, 7, 13, 19]
    current_hour = current_time.hour
    closest_hour = min(smap_hours, key=lambda x: abs(x - current_hour))
    return current_time.replace(
        hour=7, minute=30, second=0, microsecond=0
    )

def get_ifs_time_for_smap(smap_time: datetime) -> datetime:
    """Get corresponding IFS forecast time for SMAP target time."""
    smap_to_ifs = {
        1: 0,  # 01:30 uses 00:00 forecast
        7: 6,  # 07:30 uses 06:00 forecast
        13: 12,  # 13:30 uses 12:00 forecast
        19: 18,  # 19:30 uses 18:00 forecast
    }

    ifs_hour = smap_to_ifs.get(smap_time.hour)
    if ifs_hour is None:
        raise ValueError(f"Invalid SMAP time: {smap_time.hour}:30")

    return smap_time.replace(hour=ifs_hour, minute=0, second=0, microsecond=0)

def get_next_preparation_time(current_time: datetime) -> datetime:
    """Get the next preparation window start time."""
    windows = get_validator_windows()
    current_mins = current_time.hour * 60 + current_time.minute

    for start_hr, start_min, _, _ in windows:
        window_start_mins = start_hr * 60 + start_min
        if window_start_mins > current_mins:
            return current_time.replace(
                hour=start_hr, minute=start_min, second=0, microsecond=0
            )

    tomorrow = current_time + timedelta(days=1)
    first_window = windows[0]
    return tomorrow.replace(
        hour=first_window[0], minute=first_window[1], second=0, microsecond=0
    )

def get_validator_windows() -> List[Tuple[int, int, int, int]]:
    """Get all validator windows (hour_start, min_start, hour_end, min_end)."""
    return [
        (1, 30, 2, 0),  # Prep window for 1:30 SMAP time
        (2, 0, 2, 30),  # Execution window for 1:30 SMAP time
        (9, 30, 10, 0),  # Prep window for 7:30 SMAP time
        (10, 0, 10, 30),  # Execution window for 7:30 SMAP time
        (13, 30, 14, 0),  # Prep window for 13:30 SMAP time
        (14, 0, 14, 30),  # Execution window for 13:30 SMAP time
        (19, 30, 20, 0),  # Prep window for 19:30 SMAP time
        (20, 0, 20, 30),  # Execution window for 19:30 SMAP time
    ]

def _load_model(device) -> SoilModel:
    """Load model weights from local path or HuggingFace."""
    try:
        local_path = "tasks/defined_tasks/soilmoisture/SoilModel.ckpt"

        if os.path.exists(local_path):
            logger.info(f"Loading model from local path: {local_path}")
            model = SoilModel.load_from_checkpoint(local_path)
        else:
            logger.info(
                "Local checkpoint not found, downloading from HuggingFace..."
            )
            checkpoint_path = hf_hub_download(
                repo_id="Nickel5HF/soil-moisture-model",
                filename="SoilModel.ckpt",
                local_dir="tasks/defined_tasks/soilmoisture/",
            )
            logger.info(f"Loading model from HuggingFace: {checkpoint_path}")
            model = SoilModel.load_from_checkpoint(checkpoint_path)

        model.to(device)
        model.eval()

        param_count = sum(p.numel() for p in model.parameters())
        logger.info(f"Model loaded successfully with {param_count:,} parameters")
        logger.info(f"Model device: {next(model.parameters()).device}")

        return model

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        logger.error(traceback.format_exc())
        raise RuntimeError(f"Failed to load model weights: {str(e)}")
    
async def process_data(data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """Process combined tiff data for model input."""
    try:
        combined_data = data["combined_data"]
        logger.info(f"Received data type: {type(combined_data)}")
        logger.info(f"Received data: {combined_data[:100]}")

        try:
            tiff_bytes = base64.b64decode(combined_data)
        except Exception as e:
            logger.error(f"Failed to decode base64: {str(e)}")
            tiff_bytes = (
                combined_data
                if isinstance(combined_data, bytes)
                else combined_data.encode("utf-8")
            )

        logger.info(f"Decoded data size: {len(tiff_bytes)} bytes")
        logger.info(f"First 16 bytes hex: {tiff_bytes[:16].hex()}")
        logger.info(f"First 4 bytes raw: {tiff_bytes[:4]}")

        if not (
            tiff_bytes.startswith(b"II\x2A\x00")
            or tiff_bytes.startswith(b"MM\x00\x2A")
        ):
            logger.error(f"Invalid TIFF header detected")
            logger.error(f"First 16 bytes: {tiff_bytes[:16].hex()}")
            raise ValueError(
                "Invalid TIFF format: File does not start with valid TIFF header"
            )

        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".tif", delete=False, mode="wb"
            ) as temp_file:
                temp_file_path = temp_file.name
                temp_file.write(tiff_bytes)
                temp_file.flush()
                os.fsync(temp_file.fileno())

            with open(temp_file_path, "rb") as check_file:
                header = check_file.read(4)
                logger.info(f"Written file header: {header.hex()}")

            with rasterio.open(temp_file_path) as dataset:
                logger.info(
                    f"Successfully opened TIFF with shape: {dataset.shape}"
                )
                logger.info(f"TIFF metadata: {dataset.profile}")
                logger.info(
                    f"Band order: {dataset.tags().get('band_order', 'Not found')}"
                )

                model_inputs = preprocessor.preprocess(temp_file_path) # Base model
                for key in model_inputs:
                    if isinstance(model_inputs[key], torch.Tensor):
                        model_inputs[key] = model_inputs[key].to(device)
                        logger.info(f"Processed model inputs shape: { model_inputs[key].shape}")
                return model_inputs

        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except Exception as e:
                    logger.error(f"Error cleaning up temporary file: {str(e)}")

    except Exception as e:
        logger.error(f"Error processing TIFF data: {str(e)}")
        logger.error(f"Error trace: {traceback.format_exc()}")
        raise RuntimeError(f"Error processing contributor data: {str(e)}")
        
def sigmoid_rmse(rmse: float) -> float:
    """Convert RMSE to score using sigmoid function. (higher is better)"""
    alpha: float = 10
    beta: float = 0.1
    return 1 / (1 + torch.exp(alpha * (rmse - beta)))

if __name__ == "__main__":
    asyncio.run(main())