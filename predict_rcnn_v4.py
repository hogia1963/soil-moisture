import torch
import rasterio
import numpy as np
from pathlib import Path
from models.rcnn_model_v4 import RCNN_v4
import joblib
from tqdm import tqdm
from typing import Union, List, Tuple, Dict
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib.colors import CenteredNorm
from utils.data_cleaner import get_clean_data_pairs
from utils.data_organizer import DataOrganizer

class SoilMoistureV4Predictor:
    def __init__(self, checkpoint_path: str, device: str = None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = RCNN_v4(image_channels=21, history_channels=2).to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        
        # Load scalers
        self.image_scalers = [
            joblib.load(f'checkpoints/image_scaler_band_{i}_minmax.joblib')
            for i in range(21)
        ]
        self.label_scalers = [
            joblib.load(f'checkpoints/label_scaler_band_{i}_minmax.joblib')
            for i in range(2)
        ]
        
        # Load difference statistics
        try:
            stats = np.load('checkpoints/label_diff_stats.npz')
            self.diff_mean = stats['mean']
            self.diff_std = stats['std']
            print(f"Loaded difference statistics - Mean: {self.diff_mean:.6f}, Std: {self.diff_std:.6f}")
        except:
            print("Warning: Could not load difference statistics!")
            self.diff_mean = 0
            self.diff_std = 1
    
    def _transform_data(self, data: np.ndarray, scalers: List) -> np.ndarray:
        """Transform data using provided scalers"""
        transformed = np.zeros_like(data)
        for band in range(data.shape[0]):
            band_data = data[band]
            band_data = np.nan_to_num(band_data, nan=scalers[band].data_min_[0])
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
    
    def normalize_diff(self, diff):
        """Normalize difference using computed statistics"""
        return (diff - self.diff_mean) / (self.diff_std + 1e-6)
    
    def denormalize_diff(self, norm_diff):
        """Convert normalized difference back to original scale"""
        return norm_diff * self.diff_std + self.diff_mean

    def predict(self, image_t1_path: str, image_t2_path: str, label_t1_path: str) -> np.ndarray:
        """Predict soil moisture difference using two consecutive images and historical label"""
        # Load and transform data
        with rasterio.open(image_t1_path) as src:
            image_t1 = self._transform_data(src.read().astype(np.float32), self.image_scalers)
        with rasterio.open(image_t2_path) as src:
            image_t2 = self._transform_data(src.read().astype(np.float32), self.image_scalers)
        with rasterio.open(label_t1_path) as src:
            label_t1 = self._transform_data(src.read().astype(np.float32), self.label_scalers)
            
        # Convert to tensors
        with torch.no_grad():
            image_t1 = torch.FloatTensor(image_t1).unsqueeze(0).to(self.device)
            image_t2 = torch.FloatTensor(image_t2).unsqueeze(0).to(self.device)
            label_t1 = torch.FloatTensor(label_t1).unsqueeze(0).to(self.device)
            
            # Get prediction
            norm_diff_pred = self.model(image_t1, image_t2, label_t1)
            norm_diff_pred = norm_diff_pred.cpu().numpy()[0]
            
        # Denormalize prediction
        diff_pred = self.denormalize_diff(norm_diff_pred)
        
        # Calculate predicted t2 label
        with rasterio.open(label_t1_path) as src:
            label_t1_orig = src.read().astype(np.float32)
        
        label_t2_pred = label_t1_orig + diff_pred
        return label_t2_pred
    
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
    
    def predict_and_evaluate(self, image_folder: str, label_folder: str, output_dir: str = "predictions_v4"):
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
            
            for i in range(len(sequences) - 1):
                seq_t1 = sequences[i]
                seq_t2 = sequences[i + 1]
                
                image_t1_path, label_t1_path = seq_t1[-1]
                image_t2_path, label_t2_path = seq_t2[-1]
                
                try:
                    # Predict
                    prediction = self.predict(image_t1_path, image_t2_path, label_t1_path)
                    
                    # Load ground truth
                    with rasterio.open(label_t2_path) as src:
                        ground_truth = src.read().astype(np.float32)
                    
                    # Save visualization
                    output_path = output_dir / f"comparison_{Path(image_t2_path).stem}.png"
                    self.visualize_comparison(prediction, ground_truth, str(output_path))
                    
                    # Save prediction as GeoTIFF
                    output_tiff = output_dir / f"pred_{Path(image_t2_path).name}"
                    with rasterio.open(label_t2_path) as src:
                        meta = src.meta.copy()
                        with rasterio.open(output_tiff, 'w', **meta) as dst:
                            dst.write(prediction)
                            
                except Exception as e:
                    print(f"Error processing sequence: {str(e)}")
                    continue

def main():
    checkpoint_path = "checkpoints/rcnn_v4_best.pth"
    predictor = SoilMoistureV4Predictor(checkpoint_path)
    
    predictor.predict_and_evaluate(
        image_folder="data/combine_tiffs_val",
        label_folder="data/label_tiffs_val",
        output_dir="predictions_v4"
    )

if __name__ == "__main__":
    main()
