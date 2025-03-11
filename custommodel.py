# from typing import Dict, List
# import numpy as np

# class CustomSoilModel:
#     def __init__(self):
#         """Initialize your custom model"""
#         self._load_model()

#     def _load_model(self):
#         """Load your model implementation"""
#         pass

#     def run_inference(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
#         """
#         Required method that must be implemented to run model predictions.
#         run_inference should call all the methods needed to load, process data, and run the model end to end
#         Add any additional methods for data collection, preprocessing, etc. as needed.
        
#         Args:
#             inputs: Dictionary containing raw numpy arrays:
#                 sentinel_ndvi: np.ndarray[float32] [3, H, W] array with B8, B4, NDVI bands
#                 elevation: np.ndarray[float32] [1, H, W] array with elevation data
#                 era5: np.ndarray[float32] [17, H, W] array with 17 IFS weather variables

#                 reference the README or HuggingFace page for more information on the input data
#                 and weather variables

#                 Raw arrays at 500m resolution, no normalization applied (~222x222)
        
#         Returns:
#             Dictionary containing exactly:
#                 surface: list[list[float]] - 11x11 list of lists with values 0-1
#                 rootzone: list[list[float]] - 11x11 list of lists with values 0-1

#                 Must match 11x11 pixel resolution (9km resolution of SMAP L4)
                
#         Example:
#             model = CustomSoilModel()
#             predictions = model.run_inference({
#                 'sentinel_ndvi': sentinel_array,  # numpy array
#                 'elevation': elevation_array,     # numpy array
#                 'era5': era5_array               # numpy array
#             })
#         Output example: 
#             predictions = {
#                 'surface': [[0.1, 0.2, ...], [0.3, 0.4, ...], ...],  # 11x11
#                 'rootzone': [[0.2, 0.3, ...], [0.4, 0.5, ...], ...]  # 11x11
#             }
#         """
        
#         predictions:np.ndarray = {
#             'surface': np.zeros((11, 11), dtype=float),  # 11x11
#             'rootzone': np.zeros((11, 11), dtype=float)  # 11x11
#         }
#         return predictions



# """
# INPUT DATA:
# 222x222 pixels, 500m resolution, no normalization
# Some regions may have gaps in the data, check for NaNs, INFs and invalid values (negatives in SRTM)

# dict:
# {
#     "sentinel_ndvi": np.ndarray,  # shape [3, w, H]
#     "elevation": np.ndarray,      # shape [1, W, H] 
#     "era5": np.ndarray,           # shape [17, W, H]
# }
# """

import torch
import numpy as np
from CLM_LSTM_soil_moisture_prediction.ILSTM_Soil import ILSTM_SV

class CustomSoilModel:
    def __init__(self):
        self.input_dim = 20  # 17 ERA5 + 3 Sentinel features
        self.timestep = 8    
        self.output_dim = 2  
        self.hidden_size = 64
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()

    def _load_model(self):
        self.model = ILSTM_SV(
            input_dim=self.input_dim,
            timestep=self.timestep,
            output_dim=self.output_dim,
            hidden_size=self.hidden_size
        ).to(self.device)
        self.model.eval()

    def validate_inputs(self, inputs):
        required = {
            'sentinel_ndvi': (3,),
            'era5': (17,),
        }
        for key, shape in required.items():
            if key not in inputs:
                raise ValueError(f"Missing {key}")
            if inputs[key].shape[0] != shape[0]:
                raise ValueError(f"Wrong {key} shape: {inputs[key].shape}")

    def preprocess_inputs(self, inputs):
        self.validate_inputs(inputs)
        features = np.concatenate([
            inputs['era5'],
            inputs['sentinel_ndvi']
        ], axis=0)
        x = torch.FloatTensor(features).to(self.device)
        return x.unsqueeze(0)  # Add batch dimension

    def run_inference(self, inputs):
        try:
            x = self.preprocess_inputs(inputs)
            
            with torch.no_grad():
                # Get model predictions
                predictions = self.model(x)
                
                # Ensure predictions is on CPU before numpy conversion
                if isinstance(predictions, tuple):
                    predictions = predictions[0]  # Get first element if tuple
                
                # Move to CPU and convert to numpy safely
                predictions = predictions.detach().cpu().numpy()
                
                # Extract and reshape predictions
                surface = predictions[0, 0].reshape(11, 11)
                rootzone = predictions[0, 1].reshape(11, 11)
                
                # Ensure valid range
                surface = np.clip(surface, 0, 1)
                rootzone = np.clip(rootzone, 0, 1)

                return {
                    'surface': surface,
                    'rootzone': rootzone
                }

        except Exception as e:
            print(f"Inference error: {str(e)}")
            return {
                'surface': np.zeros((11, 11)),
                'rootzone': np.zeros((11, 11))
            }
