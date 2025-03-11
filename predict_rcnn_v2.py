import torch
import rasterio
import numpy as np
from pathlib import Path
from models.rcnn_model_v2 import RCNN_v2
import joblib
from tqdm import tqdm
from typing import Union, List, Tuple
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib.colors import CenteredNorm
from match_images_labels import match_images_with_labels
from utils.data_cleaner import get_clean_data_pairs
from utils.data_organizer import DataOrganizer

class SoilMoistureV2Predictor:
    def __init__(self, checkpoint_path: str, device: str = None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = RCNN_v2(image_channels=21, history_channels=2).to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.eval()
        
        # Load scalers
        self.image_scalers = [
            joblib.load(f'checkpoints/image_scaler_band_{i}.joblib')
            for i in range(21)
        ]
        self.label_scalers = [
            joblib.load(f'checkpoints/label_scaler_band_{i}.joblib')
            for i in range(2)
        ]
    
    def _transform_data(self, data: np.ndarray, scalers: List) -> np.ndarray:
        """Transform data using provided scalers"""
        transformed = np.zeros_like(data)
        for band in range(data.shape[0]):
            band_data = data[band]
            band_data = np.nan_to_num(band_data, nan=scalers[band].mean_[0])
            reshaped = band_data.reshape(-1, 1)
            transformed[band] = scalers[band].transform(reshaped).reshape(band_data.shape)
        return transformed
    
    def _inverse_transform_prediction(self, prediction: np.ndarray) -> np.ndarray:
        """Inverse transform prediction back to original scale"""
        transformed = np.zeros_like(prediction)
        for band in range(prediction.shape[0]):
            band_data = prediction[band]
            reshaped = band_data.reshape(-1, 1)
            transformed[band] = self.label_scalers[band].inverse_transform(reshaped).reshape(band_data.shape)
        return transformed

    def predict(self, current_image_path: str, historical_label_path: str) -> np.ndarray:
        """Predict soil moisture using both current image and historical label"""
        # Load and transform current image
        with rasterio.open(current_image_path) as src:
            image = src.read().astype(np.float32)
            image = self._transform_data(image, self.image_scalers)
            
        # Load and transform historical label
        with rasterio.open(historical_label_path) as src:
            historical = src.read().astype(np.float32)
            historical = self._transform_data(historical, self.label_scalers)
            
        # Convert to tensors
        with torch.no_grad():
            image_tensor = torch.FloatTensor(image).unsqueeze(0).to(self.device)
            historical_tensor = torch.FloatTensor(historical).unsqueeze(0).to(self.device)
            
            # Get prediction
            output = self.model(image_tensor, historical_tensor)
            prediction = output.cpu().numpy()[0]
            
        # Inverse transform prediction
        prediction = self._inverse_transform_prediction(prediction)
        return prediction
    
    def visualize_comparison(self, prediction: np.ndarray, ground_truth: np.ndarray, 
                           save_path: str = None):
        """Visualize prediction results compared with ground truth"""
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 3, figure=fig)
        
        # Handle NaN values
        ground_truth = np.nan_to_num(ground_truth, nan=0.0)
        prediction = np.nan_to_num(prediction, nan=0.0)
        
        # Surface moisture comparison (row 0)
        vmin_surface = min(np.nanmin(prediction[0]), np.nanmin(ground_truth[0]))
        vmax_surface = max(np.nanmax(prediction[0]), np.nanmax(ground_truth[0]))
        
        # Plot surface predictions
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(prediction[0], cmap='viridis', vmin=vmin_surface, vmax=vmax_surface)
        ax1.set_title('Predicted Surface')
        plt.colorbar(im1, ax=ax1)
        
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(ground_truth[0], cmap='viridis', vmin=vmin_surface, vmax=vmax_surface)
        ax2.set_title('Ground Truth Surface')
        plt.colorbar(im2, ax=ax2)
        
        # Surface difference
        ax3 = fig.add_subplot(gs[0, 2])
        diff_surface = prediction[0] - ground_truth[0]
        norm = CenteredNorm(vcenter=0, halfrange=np.nanmax(np.abs(diff_surface)))
        im3 = ax3.imshow(diff_surface, cmap='RdBu_r', norm=norm)
        ax3.set_title('Surface Difference')
        plt.colorbar(im3, ax=ax3)
        
        # Subsurface moisture comparison (row 1)
        vmin_subsurface = min(np.nanmin(prediction[1]), np.nanmin(ground_truth[1]))
        vmax_subsurface = max(np.nanmax(prediction[1]), np.nanmax(ground_truth[1]))
        
        ax4 = fig.add_subplot(gs[1, 0])
        im4 = ax4.imshow(prediction[1], cmap='viridis', vmin=vmin_subsurface, vmax=vmax_subsurface)
        ax4.set_title('Predicted Subsurface')
        plt.colorbar(im4, ax=ax4)
        
        ax5 = fig.add_subplot(gs[1, 1])
        im5 = ax5.imshow(ground_truth[1], cmap='viridis', vmin=vmin_subsurface, vmax=vmax_subsurface)
        ax5.set_title('Ground Truth Subsurface')
        plt.colorbar(im5, ax=ax5)
        
        # Subsurface difference
        ax6 = fig.add_subplot(gs[1, 2])
        diff_subsurface = prediction[1] - ground_truth[1]
        norm = CenteredNorm(vcenter=0, halfrange=np.nanmax(np.abs(diff_subsurface)))
        im6 = ax6.imshow(diff_subsurface, cmap='RdBu_r', norm=norm)
        ax6.set_title('Subsurface Difference')
        plt.colorbar(im6, ax=ax6)
        
        # Add metrics
        surface_mse = mean_squared_error(
            ground_truth[0][~np.isnan(ground_truth[0])],
            prediction[0][~np.isnan(ground_truth[0])]
        )
        subsurface_mse = mean_squared_error(
            ground_truth[1][~np.isnan(ground_truth[1])],
            prediction[1][~np.isnan(ground_truth[1])]
        )
        
        plt.figtext(0.02, 0.95, f'Surface MSE: {surface_mse:.4f}')
        plt.figtext(0.02, 0.45, f'Subsurface MSE: {subsurface_mse:.4f}')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    
    def predict_and_evaluate(self, image_folder: str, label_folder: str, output_dir: str = "predictions_v2"):
        """Predict and evaluate multiple images, organized by location"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Get cleaned data pairs
        matches = get_clean_data_pairs(image_folder, label_folder)
        
        if not matches:
            raise ValueError("No valid image-label pairs found!")
        
        # Organize data by location
        data_organizer = DataOrganizer(matches)
        
        # Process each location separately
        for location in data_organizer.get_locations():
            print(f"\nProcessing location {location}")
            
            # Get sequences for this location
            sequences = data_organizer.get_location_sequences(location, sequence_length=2)
            
            for sequence in tqdm(sequences, desc="Processing sequences"):
                current_image_path, current_label_path = sequence[-1]
                _, historical_label_path = sequence[-2]
                
                try:
                    # Predict
                    prediction = self.predict(current_image_path, historical_label_path)
                    
                    # Load ground truth
                    with rasterio.open(current_label_path) as src:
                        ground_truth = src.read().astype(np.float32)
                    
                    # Save visualization
                    output_path = output_dir / f"comparison_{Path(current_image_path).stem}.png"
                    self.visualize_comparison(prediction, ground_truth, str(output_path))
                    
                    # Save prediction as GeoTIFF
                    output_tiff = output_dir / f"pred_{Path(current_image_path).name}"
                    with rasterio.open(current_label_path) as src:
                        meta = src.meta.copy()
                        with rasterio.open(output_tiff, 'w', **meta) as dst:
                            dst.write(prediction)
                            
                except Exception as e:
                    print(f"Error processing {current_image_path}: {str(e)}")
                    continue

def main():
    checkpoint_path = "/mnt/e/soil-moisture/soil-moisture/checkpoints/rcnn_v3_best.pth"
    predictor = SoilMoistureV2Predictor(checkpoint_path)
    
    predictor.predict_and_evaluate(
        image_folder="data/combine_tiffs_val",
        label_folder="data/label_tiffs_val",
        output_dir="predictions_v2"
    )

if __name__ == "__main__":
    main()
