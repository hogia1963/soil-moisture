import os
import torch
import asyncio
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Union, Tuple
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm

from predict import SoilMoisturePredictor
from predict_rcnn_v2 import SoilMoistureV2Predictor 
from predict_rcnn_v3 import SoilMoistureV3Predictor
from predict_rcnn_v4 import SoilMoistureV4Predictor

# Fix import statement - remove process_smap_labels as it's a method of DataCollector
from data_collection import DataCollector, get_smap_time, get_ifs_time_for_smap
from utils.soil_apis import get_soil_data
from utils.region_selection import select_random_region
from utils.smap_api import get_smap_data

class ModelServer:
    def __init__(self, model_version: str = "v1"):
        self.model_version = model_version
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.data_collector = DataCollector()
        
        # Create data directories with improved structure
        self.data_dir = Path("serving_data")
        self.raw_dir = self.data_dir / "raw"
        
        # Create processed data directories
        self.processed_dir = self.data_dir / "processed"
        self.combine_tiff_dir = self.processed_dir / "combine_tiffs"
        self.label_tiff_dir = self.processed_dir / "label_tiffs"
        
        # Create output directories organized by model version
        self.output_dir = self.data_dir / "output" / model_version
        self.prediction_dir = self.output_dir / "predictions"
        self.viz_dir = self.output_dir / "visualizations"
        
        # Create all directories
        for dir_path in [
            self.raw_dir, 
            self.combine_tiff_dir,
            self.label_tiff_dir,
            self.prediction_dir,
            self.viz_dir
        ]:
            dir_path.mkdir(exist_ok=True, parents=True)
        
        # Initialize model based on version
        self._init_model()
        
    def _init_model(self):
        """Initialize the appropriate model version"""
        checkpoints_dir = Path("checkpoints")
        
        if self.model_version == "v1":
            checkpoint = checkpoints_dir / "rcnn_best.pth"
            self.model = SoilMoisturePredictor(str(checkpoint))
        elif self.model_version == "v2":
            checkpoint = checkpoints_dir / "rcnn_v2_best.pth"
            self.model = SoilMoistureV2Predictor(str(checkpoint))
        elif self.model_version == "v3":
            checkpoint = checkpoints_dir / "rcnn_v3_best.pth"
            self.model = SoilMoistureV3Predictor(str(checkpoint))
        elif self.model_version == "v4":
            checkpoint = checkpoints_dir / "rcnn_v4_best.pth"
            self.model = SoilMoistureV4Predictor(str(checkpoint))
        else:
            raise ValueError(f"Unsupported model version: {self.model_version}")

    async def download_data(self, target_time: datetime, bbox: List[float]) -> Tuple[str, List[float], int]:
        """Download required data for the specified time and region"""
        try:
            # Convert input time to appropriate SMAP and IFS times
            smap_time = get_smap_time(target_time)
            ifs_time = get_ifs_time_for_smap(smap_time)
            
            print(f"Target time: {target_time}")
            print(f"Converted to SMAP time: {smap_time}")
            print(f"Converted to IFS forecast time: {ifs_time}")
            
            # Create date-based subdirectories in raw
            date_str = smap_time.strftime('%Y%m%d')
            raw_subdir = self.raw_dir / date_str
            raw_subdir.mkdir(exist_ok=True)
            
            # Download data using the converted IFS time
            soil_data = await get_soil_data(bbox, ifs_time)
            if soil_data is None:
                raise RuntimeError(f"Failed to get soil data for {ifs_time}")
                
            # Move downloaded data to raw directory
            tiff_path, bounds, crs = soil_data
            if isinstance(tiff_path, str) and Path(tiff_path).exists():
                print(f"\n=== FILE TRACKING ===")
                print(f"Original file created at: {tiff_path}")
                print(f"File exists before move: {os.path.exists(tiff_path)}")
                print(f"File size: {os.path.getsize(tiff_path) / (1024*1024):.2f} MB")
                
                # Format filename using the same pattern as in data_collection.py
                bbox_str = f"{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}"
                timestamp_str = smap_time.strftime('%Y-%m-%d_%H%M')
                
                # Save raw data in raw directory with same prefix as data_collection.py
                new_tiff_path = raw_subdir / f"data_{bbox_str}_{timestamp_str}.tif"
                print(f"Moving file to: {new_tiff_path}")
                
                # Copy instead of move to keep both files
                import shutil
                shutil.copy2(tiff_path, new_tiff_path)
                print(f"File copied. Original preserved at: {tiff_path}")
                print(f"New file created at: {new_tiff_path}")
                print(f"New file exists: {os.path.exists(new_tiff_path)}")
                
                # Also copy to combine_tiff directory in processed using the same naming pattern
                processed_tiff_path = self.combine_tiff_dir / f"combined_{bbox_str}_{timestamp_str}.tif"
                shutil.copy2(tiff_path, processed_tiff_path)
                print(f"Also copied to processed directory: {processed_tiff_path}")
                
                return str(new_tiff_path), bounds, crs
            return soil_data
        except Exception as e:
            raise RuntimeError(f"Error downloading data: {str(e)}")

    async def predict_sequence(self, current_time: datetime, bbox: Optional[List[float]] = None) -> Dict:
        """Make predictions for previous, current and next time points"""
        # Get or generate bbox
        if bbox is None:
            _h3_data = self.data_collector._load_h3_map()
            _base_cells = [{"index": cell["index"], "resolution": cell["resolution"]} 
                         for cell in _h3_data["base_cells"]]
            _urban_cells = set(cell["index"] for cell in _h3_data["urban_overlay_cells"])
            _lakes_cells = set(cell["index"] for cell in _h3_data["lakes_overlay_cells"])
            bbox = select_random_region(_base_cells, _urban_cells, _lakes_cells)

        # Clean old files (optional)
        for dir_path in [self.prediction_dir, self.viz_dir, self.label_tiff_dir]:
            for file in dir_path.glob("*"):
                if file.is_file():
                    file.unlink()
        
        # Define time points with modified intervals
        time_points = [
            (current_time - timedelta(days=20), "historical"),
            (current_time - timedelta(days=18), "previous"),
            (current_time, "current")
        ]
        
        # Define which time points need SMAP labels
        label_points = ["historical", "previous"]
        
        # First download all data
        downloaded_data = {}
        
        for time_point, label in time_points:
            try:
                print(f"\n=== Downloading data for {label} time point ({time_point}) ===")
                tiff_path, bounds, crs = await self.download_data(time_point, bbox)
                downloaded_data[label] = {
                    "time": time_point,
                    "tiff_path": tiff_path,
                    "bounds": bounds,
                    "crs": crs
                }
                print(f"Successfully downloaded {label} data: {tiff_path}")
                
                # For v2/v3/v4 models, download SMAP labels for historical points
                if label in label_points:
                    try:
                        print(f"Downloading SMAP labels for {label} time point...")
                        smap_data = get_smap_data(time_point, [{"bounds": bounds, "crs": str(crs)}])
                        
                        if smap_data is not None and "region_0" in smap_data:
                            # Format the filename consistently with data_collection.py
                            bbox_str = f"{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}"
                            timestamp_str = time_point.strftime('%Y-%m-%d_%H%M')
                            label_filename = f"smap_labels_{bbox_str}_{timestamp_str}.tif"
                            label_path = self.label_tiff_dir / label_filename
                            
                            # Process SMAP data using the same function as in data_collection.py
                            processed_labels = self.data_collector.process_smap_labels(smap_data)
                            if processed_labels is None:
                                print(f"Failed to process SMAP data for {label} time point")
                                continue
                                
                            # Extract surface and rootzone soil moisture
                            surface_sm = processed_labels["surface_sm"]
                            rootzone_sm = processed_labels["rootzone_sm"]
                            
                            # Save as GeoTIFF with metadata
                            meta = {
                                'driver': 'GTiff',
                                'height': surface_sm.shape[0],
                                'width': surface_sm.shape[1],
                                'count': 2,  # Two bands: surface and rootzone
                                'dtype': np.float32,
                                'crs': crs,
                                'transform': rasterio.transform.from_bounds(
                                    bounds[0], bounds[1], bounds[2], bounds[3],
                                    surface_sm.shape[1], surface_sm.shape[0]
                                )
                            }
                            
                            with rasterio.open(label_path, 'w', **meta) as dst:
                                # Write surface soil moisture as band 1
                                dst.write(surface_sm.astype(np.float32), 1)
                                # Write rootzone soil moisture as band 2
                                dst.write(rootzone_sm.astype(np.float32), 2)
                                # Add descriptions for bands
                                dst.update_tags(1, DESCRIPTION="Surface Soil Moisture")
                                dst.update_tags(2, DESCRIPTION="Rootzone Soil Moisture")
                            
                            print(f"Saved SMAP labels to: {label_path}")
                            downloaded_data[label]["label_path"] = label_path
                        else:
                            print(f"No SMAP data available for {label} time point")
                            print(f"Debug info - smap_data returned: {smap_data}")
                            # Create dummy/synthetic labels when SMAP data isn't available for testing
                            if self.model_version in ["v2", "v3", "v4"]:
                                print(f"Creating synthetic labels for {label} for testing purposes")
                                # Open the image file to get metadata
                                with rasterio.open(tiff_path) as src:
                                    # Create dummy data of appropriate shape
                                    height = 11  # Standard SMAP grid size
                                    width = 11   # Standard SMAP grid size
                                    
                                    # Create synthetic data with reasonable moisture values
                                    surface_sm = np.ones((height, width), dtype=np.float32) * 0.3  # 30% moisture
                                    rootzone_sm = np.ones((height, width), dtype=np.float32) * 0.25  # 25% moisture
                                    
                                    # Add some variation
                                    surface_sm += np.random.normal(0, 0.05, size=(height, width))
                                    rootzone_sm += np.random.normal(0, 0.03, size=(height, width))
                                    
                                    # Clip to valid moisture range (0-1)
                                    surface_sm = np.clip(surface_sm, 0.05, 0.6)
                                    rootzone_sm = np.clip(rootzone_sm, 0.05, 0.5)
                                    
                                    # Format filename for synthetic data with consistent prefix
                                    bbox_str = f"{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}"
                                    timestamp_str = time_point.strftime('%Y-%m-%d_%H%M')
                                    label_filename = f"synthetic_smap_labels_{bbox_str}_{timestamp_str}.tif"
                                    label_path = self.label_tiff_dir / label_filename
                                    
                                    meta = {
                                        'driver': 'GTiff',
                                        'height': height,
                                        'width': width,
                                        'count': 2,
                                        'dtype': np.float32,
                                        'crs': crs,
                                        'transform': rasterio.transform.from_bounds(
                                            bounds[0], bounds[1], bounds[2], bounds[3], width, height
                                        )
                                    }
                                    
                                    with rasterio.open(label_path, 'w', **meta) as dst:
                                        dst.write(surface_sm.reshape(1, *surface_sm.shape), 1)
                                        dst.write(rootzone_sm.reshape(1, *rootzone_sm.shape), 2)
                                        dst.update_tags(1, DESCRIPTION="Surface Soil Moisture (Synthetic)")
                                        dst.update_tags(2, DESCRIPTION="Rootzone Soil Moisture (Synthetic)")
                                    
                                    print(f"Saved synthetic SMAP labels to: {label_path}")
                                    downloaded_data[label]["label_path"] = label_path
                                    downloaded_data[label]["is_synthetic"] = True
                    except Exception as e:
                        print(f"Error downloading SMAP labels for {label}: {str(e)}")
                        print("Traceback:")
                        import traceback
                        print(traceback.format_exc())
            except Exception as e:
                print(f"Error downloading {label} data: {str(e)}")
                # Continue with other time points
        
        # Now process predictions based on available data
        results = {}
        
        # First predict for "previous" (5 days ago) using "historical" (6 days ago)
        if "historical" in downloaded_data and "previous" in downloaded_data:
            try:
                print("\n=== Predicting for 5 days ago using data from 6 days ago ===")
                
                # Get historical data
                historical_data = downloaded_data["historical"]
                historical_path = historical_data["tiff_path"]
                historical_label_path = historical_data.get("label_path")
                historical_is_synthetic = historical_data.get("is_synthetic", False)
                
                # Get previous data for ground truth comparison
                previous_data = downloaded_data["previous"]
                previous_path = previous_data["tiff_path"]
                previous_label_path = previous_data.get("label_path")
                previous_is_synthetic = previous_data.get("is_synthetic", False)
                
                bounds = previous_data["bounds"]
                crs = previous_data["crs"]
                time_point = previous_data["time"]
                
                # Check if we have the required data
                has_required_data = historical_path and os.path.exists(historical_path)
                has_historical_label = historical_label_path and os.path.exists(historical_label_path)
                has_previous_label = previous_label_path and os.path.exists(previous_label_path)
                
                if has_required_data:
                    # Decide which prediction method to use based on available data
                    if has_historical_label:
                        print(f"Using historical labels with {'synthetic' if historical_is_synthetic else 'real'} data")
                        
                        # Different prediction methods based on model version
                        if self.model_version == "v3":
                            # V3 model requires current_image_path, previous_image_path, and historical_label_path
                            prediction = self.model.predict(
                                current_image_path=previous_path,
                                previous_image_path=historical_path,
                                historical_label_path=historical_label_path
                            )
                        elif self.model_version == "v2":
                            # V2 model requires current_image_path and historical_label_path
                            prediction = self.model.predict(previous_path, historical_label_path)
                        elif self.model_version == "v4":
                            # V4 model requires image_t1_path, image_t2_path, and label_t1_path
                            prediction = self.model.predict(historical_path, previous_path, historical_label_path)
                        else:
                            # V1 model only requires image_path
                            prediction = self.model.predict(previous_path)
                        
                        # Save prediction to output/predictions directory with consistent naming
                        bbox_str = f"{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}"
                        timestamp_str = time_point.strftime('%Y-%m-%d_%H%M')
                        pred_filename = f"prediction_previous_{bbox_str}_{timestamp_str}.tif"
                        pred_path = self.prediction_dir / pred_filename
                        
                        # Get metadata from the label to ensure consistent CRS and transform
                        with rasterio.open(previous_label_path if has_previous_label else historical_label_path) as label_src:
                            label_meta = label_src.meta.copy()
                            
                            # Use the label's transform and CRS
                            meta = {
                                'driver': 'GTiff',
                                'height': 11,  # Standard SMAP grid size
                                'width': 11,   # Standard SMAP grid size
                                'count': 2,    # 2 channels: surface and rootzone
                                'dtype': np.float32,
                                'crs': label_meta['crs'],
                                'transform': label_meta['transform']
                            }
                            
                            # Ensure prediction has correct shape (11x11) for SMAP grid
                            if prediction.shape[1:] != (11, 11):
                                print(f"Reshaping prediction from {prediction.shape} to [2, 11, 11]")
                                # Reshape prediction to match SMAP grid
                                if len(prediction.shape) == 3:  # shape is [2, H, W]
                                    prediction = np.array([
                                        np.resize(prediction[0], (11, 11)),
                                        np.resize(prediction[1], (11, 11))
                                    ])
                                else:  # shape is [H, W]
                                    prediction = np.resize(prediction, (1, 11, 11))
                            
                            with rasterio.open(pred_path, 'w', **meta) as dst:
                                dst.write(prediction)
                        
                        print(f"Saved prediction to: {pred_path}")
                        
                        # Compare with ground truth if available
                        if has_previous_label:
                            # Load ground truth from the previous label for comparison
                            with rasterio.open(previous_label_path) as src:
                                ground_truth = src.read()
                            
                            # Save comparison visualization to output/visualizations with consistent naming
                            viz_path = self.viz_dir / f"comparison_previous_{bbox_str}_{timestamp_str}.png"
                            self.model.visualize_comparison(prediction, ground_truth, str(viz_path))
                            print(f"Saved comparison visualization to: {viz_path}")
                            
                            # Store results
                            results["previous"] = {
                                "time": time_point,
                                "prediction": prediction,
                                "prediction_path": pred_path,
                                "ground_truth": ground_truth,
                                "ground_truth_path": previous_label_path,
                                "ground_truth_is_synthetic": previous_is_synthetic,
                                "tiff_path": previous_path,
                                "bounds": bounds,
                                "crs": crs
                            }
                        else:
                            # No ground truth for comparison
                            results["previous"] = {
                                "time": time_point,
                                "prediction": prediction,
                                "prediction_path": pred_path,
                                "ground_truth": None,
                                "tiff_path": previous_path,
                                "bounds": bounds,
                                "crs": crs
                            }
                    else:
                        print("Cannot predict for 'previous' - required historical labels not available")
                        print("Available data for historical:", historical_path)
                        print("Available label for historical:", historical_label_path)
                else:
                    print("Cannot predict for 'previous' - required historical data not available")
            except Exception as e:
                print(f"Error processing previous prediction: {str(e)}")
                import traceback
                print(traceback.format_exc())
                
        # Now predict for "current" using "previous" (5 days ago) and current image
        if "previous" in downloaded_data and "current" in downloaded_data:
            try:
                print("\n=== Predicting for current time using data from 5 days ago and current image ===")
                
                # Get previous data
                previous_data = downloaded_data["previous"]
                previous_path = previous_data["tiff_path"]
                previous_label_path = previous_data.get("label_path")
                previous_is_synthetic = previous_data.get("is_synthetic", False)
                
                # Get current data
                current_data = downloaded_data["current"]
                current_path = current_data["tiff_path"]
                bounds = current_data["bounds"]
                crs = current_data["crs"]
                time_point = current_data["time"]
                
                # Check if we have the required data
                has_required_data = previous_path and os.path.exists(previous_path) and current_path and os.path.exists(current_path)
                has_previous_label = previous_label_path and os.path.exists(previous_label_path)
                
                if has_required_data and has_previous_label:
                    print(f"Using previous labels with {'synthetic' if previous_is_synthetic else 'real'} data")
                    
                    # Different prediction methods based on model version
                    if self.model_version == "v3":
                        # V3 model requires current_image_path, previous_image_path, and historical_label_path
                        prediction = self.model.predict(
                            current_image_path=current_path,
                            previous_image_path=previous_path,
                            historical_label_path=previous_label_path
                        )
                    elif self.model_version == "v2":
                        # V2 model requires current_image_path and historical_label_path
                        prediction = self.model.predict(current_path, previous_label_path)
                    elif self.model_version == "v4":
                        # V4 model requires image_t1_path, image_t2_path, and label_t1_path
                        prediction = self.model.predict(previous_path, current_path, previous_label_path)
                    else:
                        # V1 model only requires image_path
                        prediction = self.model.predict(current_path)
                    
                    # Save prediction to output/predictions directory with consistent naming
                    bbox_str = f"{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}"
                    timestamp_str = time_point.strftime('%Y-%m-%d_%H%M')
                    pred_filename = f"prediction_current_{bbox_str}_{timestamp_str}.tif"
                    pred_path = self.prediction_dir / pred_filename
                    
                    # Get metadata from the label to ensure consistent CRS and transform
                    with rasterio.open(previous_label_path) as label_src:
                        label_meta = label_src.meta.copy()
                        
                        # Use the label's transform and CRS
                        meta = {
                            'driver': 'GTiff',
                            'height': 11,  # Standard SMAP grid size
                            'width': 11,   # Standard SMAP grid size
                            'count': 2,    # 2 channels: surface and rootzone
                            'dtype': np.float32,
                            'crs': label_meta['crs'],
                            'transform': label_meta['transform']
                        }
                        
                        # Ensure prediction has correct shape (11x11) for SMAP grid
                        if prediction.shape[1:] != (11, 11):
                            print(f"Reshaping prediction from {prediction.shape} to [2, 11, 11]")
                            # Reshape prediction to match SMAP grid
                            if len(prediction.shape) == 3:  # shape is [2, H, W]
                                prediction = np.array([
                                    np.resize(prediction[0], (11, 11)),
                                    np.resize(prediction[1], (11, 11))
                                ])
                            else:  # shape is [H, W]
                                prediction = np.resize(prediction, (1, 11, 11))
                        
                        with rasterio.open(pred_path, 'w', **meta) as dst:
                            dst.write(prediction)
                    
                    print(f"Saved prediction to: {pred_path}")
                    
                    # Get SMAP ground truth if available for current time
                    try:
                        smap_data = get_smap_data(time_point, [{"bounds": bounds, "crs": str(crs)}])
                        ground_truth = None
                        if smap_data and "region_0" in smap_data:
                            ground_truth = smap_data["region_0"]
                        
                        # Visualize if ground truth is available
                        if ground_truth is not None:
                            # Extract data from ground_truth for visualization
                            surface_sm = ground_truth["surface_sm"]
                            rootzone_sm = ground_truth["rootzone_sm"]
                            ground_truth_array = np.stack([surface_sm, rootzone_sm])
                            
                            # Save visualization to output/visualizations with consistent naming
                            viz_path = self.viz_dir / f"comparison_current_{bbox_str}_{timestamp_str}.png"
                            self.model.visualize_comparison(prediction, ground_truth_array, str(viz_path))
                            print(f"Saved comparison visualization to: {viz_path}")
                    except Exception as e:
                        print(f"Error getting ground truth for current time: {str(e)}")
                        ground_truth = None
                    
                    # Store results
                    results["current"] = {
                        "time": time_point,
                        "prediction": prediction,
                        "prediction_path": pred_path,
                        "ground_truth": ground_truth,
                        "tiff_path": current_path,
                        "bounds": bounds,
                        "crs": crs
                    }
                else:
                    print("Cannot predict for 'current' - required previous labels not available")
                    print("Available data for previous:", previous_path)
                    print("Available label for previous:", previous_label_path)
                    print("Available data for current:", current_path)
            except Exception as e:
                print(f"Error processing current prediction: {str(e)}")
                import traceback
                print(traceback.format_exc())

        return results

def main():
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description='Serve soil moisture predictions')
    parser.add_argument('--model', type=str, default='v1', choices=['v1', 'v2', 'v3', 'v4'],
                      help='Model version to use')
    parser.add_argument('--time', type=str, help='Target time in ISO format. If not provided, uses current time')
    parser.add_argument('--bbox', type=float, nargs=4, help='Bounding box coordinates: minlon minlat maxlon maxlat')
    parser.add_argument('--keep-data', action='store_true', help='Keep downloaded data files')
    
    args = parser.parse_args()

    # Initialize server
    server = ModelServer(model_version=args.model)
    
    # Set up time and bbox
    if args.time:
        current_time = datetime.fromisoformat(args.time).replace(tzinfo=timezone.utc)
    else:
        current_time = datetime.now(timezone.utc)
        
    bbox = args.bbox if args.bbox else None
    if bbox:
        bbox = list(bbox)

    # Run predictions
    try:
        results = asyncio.run(server.predict_sequence(current_time, bbox))
        print("\nPrediction Results:")
        for time_point, data in results.items():
            print(f"\n{time_point.upper()}:")
            print(f"Time: {data['time']}")
            print(f"Bounds: {data['bounds']}")
            print(f"Prediction saved to: {data['prediction_path']}")
            print(f"Raw data: {data['tiff_path']}")
            if data['ground_truth'] is not None:
                print("Ground truth comparison visualization saved")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
