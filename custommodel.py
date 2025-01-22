from typing import Dict, List
import numpy as np

class CustomSoilModel:
    def __init__(self):
        """Initialize your custom model"""
        self._load_model()

    def _load_model(self):
        """Load your model implementation"""
        pass

    def run_inference(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Required method that must be implemented to run model predictions.
        run_inference should call all the methods needed to load, process data, and run the model end to end
        Add any additional methods for data collection, preprocessing, etc. as needed.
        
        Args:
            inputs: Dictionary containing raw numpy arrays:
                sentinel_ndvi: np.ndarray[float32] [3, H, W] array with B8, B4, NDVI bands
                elevation: np.ndarray[float32] [1, H, W] array with elevation data
                era5: np.ndarray[float32] [17, H, W] array with 17 IFS weather variables

                reference the README or HuggingFace page for more information on the input data
                and weather variables

                Raw arrays at 500m resolution, no normalization applied (~222x222)
        
        Returns:
            Dictionary containing exactly:
                surface: list[list[float]] - 11x11 list of lists with values 0-1
                rootzone: list[list[float]] - 11x11 list of lists with values 0-1

                Must match 11x11 pixel resolution (9km resolution of SMAP L4)
                
        Example:
            model = CustomSoilModel()
            predictions = model.run_inference({
                'sentinel_ndvi': sentinel_array,  # numpy array
                'elevation': elevation_array,     # numpy array
                'era5': era5_array               # numpy array
            })
        Output example: 
            predictions = {
                'surface': [[0.1, 0.2, ...], [0.3, 0.4, ...], ...],  # 11x11
                'rootzone': [[0.2, 0.3, ...], [0.4, 0.5, ...], ...]  # 11x11
            }
        """
        
        predictions:np.ndarray = {
            'surface': np.zeros((11, 11), dtype=float),  # 11x11
            'rootzone': np.zeros((11, 11), dtype=float)  # 11x11
        }
        return predictions



"""
INPUT DATA:
222x222 pixels, 500m resolution, no normalization
Some regions may have gaps in the data, check for NaNs, INFs and invalid values (negatives in SRTM)

dict:
{
    "sentinel_ndvi": np.ndarray,  # shape [3, w, H]
    "elevation": np.ndarray,      # shape [1, W, H] 
    "era5": np.ndarray,           # shape [17, W, H]
}
"""