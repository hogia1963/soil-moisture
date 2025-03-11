import os
import asyncio
from datetime import datetime, timedelta, timezone
import numpy as np
from typing import Dict, List, Tuple
import torch
from utils.soil_apis import get_soil_data
from utils.region_selection import select_random_region
from utils.smap_api import get_smap_data
from utils.inference_class import SoilMoistureInferencePreprocessor
import json
import rasterio

def get_smap_time(current_time: datetime) -> datetime:
    """Get SMAP time based on validator execution time."""
    smap_hours = [1, 7, 13, 19]
    current_hour = current_time.hour
    closest_hour = min(smap_hours, key=lambda x: abs(x - current_hour))
    return current_time.replace(
        hour=closest_hour, minute=30, second=0, microsecond=0
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

class DataCollector:
    """Collect and process training data with corresponding SMAP labels."""
    
    def __init__(self, data_dir: str = None):
        """Initialize data collector."""
        if data_dir is None:
            data_dir = os.path.join(os.getcwd(), "data")
        os.makedirs(data_dir, exist_ok=True)
        self.data_dir = data_dir
        self.preprocessor = SoilMoistureInferencePreprocessor()
        
        # Load H3 map data
        self._h3_data = self._load_h3_map()
        self._base_cells = [
            {"index": cell["index"], "resolution": cell["resolution"]}
            for cell in self._h3_data["base_cells"]
        ]
        self._urban_cells = set(cell["index"] for cell in self._h3_data["urban_overlay_cells"])
        self._lakes_cells = set(cell["index"] for cell in self._h3_data["lakes_overlay_cells"])

    def _load_h3_map(self) -> dict:
            """Load H3 map data from local file or HuggingFace."""
            try:
                local_path = "./data/h3_map/full_h3_map.json"
                if os.path.exists(local_path):
                    with open(local_path, "r") as f:
                        return json.load(f)
                
                from huggingface_hub import hf_hub_download
                map_path = hf_hub_download(
                    repo_id="Nickel5HF/gaia_h3_mapping",
                    filename="full_h3_map.json",
                    repo_type="dataset",
                    local_dir="./data/h3_map",
                )
                with open(map_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                raise RuntimeError(f"Failed to load H3 map: {str(e)}")

    def process_tiff_data(self, tiff_path: str) -> Dict[str, np.ndarray]:
        """Process TIFF file to extract features."""
        try:
            with rasterio.open(tiff_path) as src:
                # Read and organize bands according to their purpose
                sentinel_data = src.read([1, 2])  # B8, B4 bands
                ifs_data = src.read(list(range(3, 20)))  # IFS variables
                elevation = src.read([20])  # SRTM
                ndvi = src.read([21])  # NDVI

                # Combine Sentinel bands with NDVI
                sentinel_ndvi = np.vstack([sentinel_data, ndvi])  # [3, H, W]

                return {
                    "sentinel_ndvi": sentinel_ndvi,
                    "elevation": elevation,
                    "era5": ifs_data
                }
        except Exception as e:
            print(f"Error processing TIFF file: {str(e)}")
            return None

    def process_smap_labels(self, smap_data: Dict) -> Dict[str, np.ndarray]:
        """Process SMAP labels to correct format."""
        try:
            surface_sm = np.array(smap_data["region_0"]["surface_sm"])
            rootzone_sm = np.array(smap_data["region_0"]["rootzone_sm"])
            
            # Ensure labels are in correct shape (11x11)
            if surface_sm.shape != (11, 11) or rootzone_sm.shape != (11, 11):
                surface_sm = np.resize(surface_sm, (11, 11))
                rootzone_sm = np.resize(rootzone_sm, (11, 11))
            
            return {
                "surface_sm": surface_sm,
                "rootzone_sm": rootzone_sm
            }
        except Exception as e:
            print(f"Error processing SMAP labels: {str(e)}")
            return None

    async def collect_training_data(self, num_days: int = 10, days_ago: int = 30) -> List[Dict]:
        """Collect and process training data with labels."""
        current_time = datetime.now(timezone.utc)
        collected_data = []
        for i in range(15):
            bbox = select_random_region(
                    base_cells=self._base_cells,
                    urban_cells_set=self._urban_cells,
                    lakes_cells_set=self._lakes_cells,
                        )
            for day in range(days_ago + num_days,days_ago, -1):
                target_date = current_time - timedelta(days=day)
                
                print(f"\nCollecting data for {target_date.date()}")
                for hour in [1,7,13,19]:  #[0, 3, 6, 9, 12, 15, 18, 21]:
                    timestamp = target_date.replace(hour=hour)
                    target_smap_time = get_smap_time(timestamp)
                    ifs_forecast_time = get_ifs_time_for_smap(target_smap_time)
                    try:
                        # Get random region
                        
                        # Get input features
                        soil_data = await get_soil_data(bbox, ifs_forecast_time)
                        if soil_data is None:
                            print(f"Failed to get soil data for {ifs_forecast_time.date()}")
                            continue
                            
                        tiff_path, bounds, crs = soil_data
                        
                        # Process input features
                        processed_inputs = self.process_tiff_data(tiff_path)
                        if processed_inputs is None:
                            print(f"Failed to process input data for {ifs_forecast_time.date()}")
                            continue

                        # Get SMAP labels
                        regions = [{
                            "bounds": bounds,
                            "crs": str(crs)
                        }]
                        smap_data = get_smap_data(target_smap_time, regions)
                        
                        if smap_data is None:
                            print(f"Failed to get SMAP data for {target_smap_time.date()}")
                            continue

                        # Process SMAP labels
                        processed_labels = self.process_smap_labels(smap_data)
                        if processed_labels is None:
                            print(f"Failed to process SMAP labels for {target_smap_time.date()}")
                            continue
                        labels_dir = os.path.join(self.data_dir, "label_tiffs")
                        os.makedirs(labels_dir, exist_ok=True)
                        
                        # Create GeoTIFF for surface and rootzone soil moisture
                        tiff_metadata = {
                            'driver': 'GTiff',
                            'height': processed_labels["surface_sm"].shape[0],
                            'width': processed_labels["surface_sm"].shape[1],
                            'count': 2,  # Two bands: surface and rootzone
                            'dtype': np.float32,
                            'crs': crs,
                            'transform': rasterio.transform.from_bounds(
                                bounds[0], bounds[1], bounds[2], bounds[3],
                                processed_labels["surface_sm"].shape[1],
                                processed_labels["surface_sm"].shape[0]
                            )
                        }
                        
                        label_tiff_path = os.path.join(
                            labels_dir,
                            f"smap_labels_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}_{target_smap_time.strftime('%Y-%m-%d_%H%M')}.tiff"
                        )
                        
                        with rasterio.open(label_tiff_path, 'w', **tiff_metadata) as dst:
                            # Write surface soil moisture as band 1
                            dst.write(processed_labels["surface_sm"].astype(np.float32), 1)
                            # Write rootzone soil moisture as band 2
                            dst.write(processed_labels["rootzone_sm"].astype(np.float32), 2)
                            
                            # Add descriptions for bands
                            dst.update_tags(1, DESCRIPTION="Surface Soil Moisture")
                            dst.update_tags(2, DESCRIPTION="Rootzone Soil Moisture")
                        # Create sample with processed data
                    # sample = {
                    #     "date": timestamp.isoformat(),
                    #     "bbox": bbox,
                    #     "bounds": bounds,
                    #     "crs": str(crs),
                    #     "inputs": {
                    #         "sentinel_ndvi": processed_inputs["sentinel_ndvi"].tolist(),
                    #         "elevation": processed_inputs["elevation"].tolist(),
                    #         "era5": processed_inputs["era5"].tolist()
                    #     },
                    #     "labels": {
                    #         "surface_sm": processed_labels["surface_sm"].tolist(),
                    #         "rootzone_sm": processed_labels["rootzone_sm"].tolist()
                    #     }
                    # }
                    
                    # collected_data.append(sample)
                    
                    # # Save to disk
                    # sample_path = os.path.join(
                    #     self.data_dir, 
                    #     f"processed_sample_{timestamp.strftime('%Y%m%d')}.json"
                    # )
                    # with open(sample_path, "w") as f:
                    #     json.dump(sample, f)
                    
                        print(f"Successfully processed and saved data for {target_smap_time.date()}")
                    
                    except Exception as e:
                        print(f"Error processing data for {target_smap_time.date()}: {str(e)}")
                    continue
                    
            # return collected_data

    def load_training_data(self, start_date: datetime = None, end_date: datetime = None) -> List[Dict]:
        """Load collected training data from disk.
        
        Args:
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            List of training samples
        """
        samples = []
        
        for filename in os.listdir(self.data_dir):
            if not filename.endswith(".json"):
                continue
                
            try:
                with open(os.path.join(self.data_dir, filename), "r") as f:
                    sample = json.load(f)
                
                sample_date = datetime.fromisoformat(sample["date"])
                
                if start_date and sample_date < start_date:
                    continue
                if end_date and sample_date > end_date:
                    continue
                    
                samples.append(sample)
                
            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")
                continue
                
        return samples

# Example usage
async def main():
    # Initialize collector
    collector = DataCollector()
    
    # Collect and process 5 days of training data
    processed_data = await collector.collect_training_data(num_days=2, days_ago=210)
    
    # Load processed data
    start_date = datetime.now(timezone.utc) - timedelta(days=30)
    end_date = datetime.now(timezone.utc) - timedelta(days=25)
    loaded_data = collector.load_training_data(start_date, end_date)
    
    print(f"Processed {len(processed_data)} new samples")
    print(f"Loaded {len(loaded_data)} existing samples")

    # Print data statistics
    if processed_data:
        sample = processed_data[0]
        print("\nSample data structure:")
        print(f"Date: {sample['date']}")
        print(f"Input shapes:")
        print(f"- Sentinel-NDVI: {np.array(sample['inputs']['sentinel_ndvi']).shape}")
        print(f"- Elevation: {np.array(sample['inputs']['elevation']).shape}")
        print(f"- ERA5: {np.array(sample['inputs']['era5']).shape}")
        print(f"Label shapes:")
        print(f"- Surface SM: {np.array(sample['labels']['surface_sm']).shape}")
        print(f"- Rootzone SM: {np.array(sample['labels']['rootzone_sm']).shape}")


if __name__ == "__main__":
    asyncio.run(main())