import torch
import rasterio
import numpy as np
from pathlib import Path
from models.rcnn_model import RCNN
import joblib
from tqdm import tqdm
from typing import Union, List
import matplotlib.pyplot as plt
from match_images_labels import match_images_with_labels
import matplotlib.gridspec as gridspec
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib.colors import CenteredNorm
from utils.data_cleaner import get_clean_data_pairs  # Add this import

class SoilMoisturePredictor:
    def __init__(self, checkpoint_path: str, device: str = None):
        # Setup device
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = RCNN(input_channels=21).to(self.device)
        
        # Load checkpoint with more robust error handling
        print(f"Loading checkpoint from {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            
            # Handle both old and new checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    print("Loading new checkpoint format...")
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    print("Loading checkpoint as direct state dict...")
                    self.model.load_state_dict(checkpoint)
            else:
                print("Loading legacy checkpoint format...")
                self.model.load_state_dict(checkpoint)
        except Exception as e:
            raise RuntimeError(f"Error loading checkpoint: {str(e)}")
            
        self.model.eval()
        
        # Load both image and label scalers with new minmax suffix
        self.image_scalers = [
            joblib.load(f'checkpoints/image_scaler_band_{i}_minmax.joblib')
            for i in range(21)
        ]
        self.label_scalers = [
            joblib.load(f'checkpoints/label_scaler_band_{i}_minmax.joblib')
            for i in range(2)
        ]
    
    def _transform_image(self, image: np.ndarray) -> np.ndarray:
        """Transform image using saved scalers"""
        transformed = np.zeros_like(image)
        for band in range(image.shape[0]):
            band_data = image[band]
            band_data = np.nan_to_num(band_data, nan=self.image_scalers[band].data_min_[0])
            reshaped = band_data.reshape(-1, 1)
            transformed[band] = self.image_scalers[band].transform(reshaped).reshape(band_data.shape)
        return transformed
    
    def _inverse_transform_prediction(self, prediction: np.ndarray) -> np.ndarray:
        """Inverse transform prediction back to original scale"""
        transformed = np.zeros_like(prediction)
        for band in range(prediction.shape[0]):
            band_data = prediction[band]
            reshaped = band_data.reshape(-1, 1)
            transformed[band] = self.label_scalers[band].inverse_transform(reshaped).reshape(band_data.shape)
        return transformed

    def predict(self, image_path: Union[str, Path]) -> np.ndarray:
        """Predict soil moisture for a single image"""
        # Load and transform image
        with rasterio.open(image_path) as src:
            image = src.read().astype(np.float32)
            image = self._transform_image(image)
            
        # Convert to tensor and predict
        with torch.no_grad():
            image_tensor = torch.FloatTensor(image).unsqueeze(0).to(self.device)
            output = self.model(image_tensor)
            prediction = output.cpu().numpy()[0]  # Shape: (2, 11, 11)
            
        # Inverse transform prediction
        prediction = self._inverse_transform_prediction(prediction)
        return prediction
    
    def predict_batch(self, image_paths: List[Union[str, Path]], output_dir: str = "predictions"):
        """Predict soil moisture for multiple images and save results"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        for image_path in tqdm(image_paths, desc="Predicting"):
            # Predict
            prediction = self.predict(image_path)
            
            # Save prediction
            output_path = output_dir / f"pred_{Path(image_path).name}"
            with rasterio.open(image_path) as src:
                # Get metadata from input image
                meta = src.meta.copy()
                meta.update({
                    'count': 2,  # 2 channels output
                    'height': 11,
                    'width': 11
                })
                
                # Save prediction as GeoTIFF
                with rasterio.open(output_path, 'w', **meta) as dst:
                    dst.write(prediction)
    
    def visualize_prediction(self, prediction: np.ndarray, save_path: str = None):
        """Visualize prediction results"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        
        # Plot first channel
        im1 = ax1.imshow(prediction[0], cmap='viridis')
        ax1.set_title('Surface Soil Moisture')
        plt.colorbar(im1, ax=ax1)
        
        # Plot second channel
        im2 = ax2.imshow(prediction[1], cmap='viridis')
        ax2.set_title('Subsurface Soil Moisture')
        plt.colorbar(im2, ax=ax2)
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def visualize_comparison(self, prediction: np.ndarray, label_path: str, save_path: str = None):
        """Visualize prediction results compared with ground truth"""
        # Load ground truth and handle NaN values
        with rasterio.open(label_path) as src:
            ground_truth = src.read().astype(np.float32)
            ground_truth = np.nan_to_num(ground_truth, nan=0.0)
        
        # Handle NaN in prediction
        prediction = np.nan_to_num(prediction, nan=0.0)
        
        # Create masks for valid (non-NaN) values in original data
        valid_mask_surface = ~np.isnan(ground_truth[0])
        valid_mask_subsurface = ~np.isnan(ground_truth[1])
        
        # Create figure with GridSpec
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 3, figure=fig)
        
        # Surface moisture comparison (row 0)
        vmin_surface = min(np.nanmin(prediction[0]), np.nanmin(ground_truth[0]))
        vmax_surface = max(np.nanmax(prediction[0]), np.nanmax(ground_truth[0]))
        
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(prediction[0], cmap='viridis', vmin=vmin_surface, vmax=vmax_surface)
        ax1.set_title('Predicted Surface')
        plt.colorbar(im1, ax=ax1)
        
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(ground_truth[0], cmap='viridis', vmin=vmin_surface, vmax=vmax_surface)
        ax2.set_title('Ground Truth Surface')
        plt.colorbar(im2, ax=ax2)
        
        ax3 = fig.add_subplot(gs[0, 2])
        diff_surface = np.where(valid_mask_surface, prediction[0] - ground_truth[0], np.nan)
        max_abs_diff = np.nanmax(np.abs(diff_surface))
        norm = CenteredNorm(vcenter=0, halfrange=max_abs_diff)
        im3 = ax3.imshow(diff_surface, cmap='RdBu_r', norm=norm)
        ax3.set_title('Difference (Pred - GT)')
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
        
        ax6 = fig.add_subplot(gs[1, 2])
        diff_subsurface = np.where(valid_mask_subsurface, prediction[1] - ground_truth[1], np.nan)
        max_abs_diff = np.nanmax(np.abs(diff_subsurface))
        norm = CenteredNorm(vcenter=0, halfrange=max_abs_diff)
        im6 = ax6.imshow(diff_subsurface, cmap='RdBu_r', norm=norm)
        ax6.set_title('Difference (Pred - GT)')
        plt.colorbar(im6, ax=ax6)
        
        # Calculate metrics only on valid values
        surface_mse = mean_squared_error(
            ground_truth[0][valid_mask_surface],
            prediction[0][valid_mask_surface]
        )
        subsurface_mse = mean_squared_error(
            ground_truth[1][valid_mask_subsurface],
            prediction[1][valid_mask_subsurface]
        )
        surface_r2 = r2_score(
            ground_truth[0][valid_mask_surface],
            prediction[0][valid_mask_surface]
        )
        subsurface_r2 = r2_score(
            ground_truth[1][valid_mask_subsurface],
            prediction[1][valid_mask_subsurface]
        )
        
        plt.figtext(0.02, 0.95, f'Surface - MSE: {surface_mse:.4f}, R²: {surface_r2:.4f}')
        plt.figtext(0.02, 0.45, f'Subsurface - MSE: {subsurface_mse:.4f}, R²: {subsurface_r2:.4f}')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
    def predict_and_compare(self, image_paths: List[Union[str, Path]], label_folder: str, output_dir: str = "predictions"):
        """Predict and compare with ground truth for multiple images"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Get cleaned data pairs
        print("Cleaning data pairs...")
        image_folder = str(Path(image_paths[0]).parent)
        matches = get_clean_data_pairs(
            image_folder=image_folder,
            label_folder=label_folder,
            max_nan_percentage=50.0,
            save_report=True
        )
        
        if not matches:
            raise ValueError("No valid image-label pairs found after cleaning!")
        
        print(f"Found {len(matches)} clean image-label pairs")
        
        # Filter image_paths to only include cleaned images
        clean_image_paths = [path for path in image_paths if str(path) in matches]
        if len(clean_image_paths) == 0:
            raise ValueError("No clean images found in provided image paths!")
        
        print(f"Processing {len(clean_image_paths)} clean images")
        
        for image_path in tqdm(clean_image_paths, desc="Predicting and comparing"):
            image_path_str = str(image_path)
            if image_path_str in matches:
                try:
                    # Predict
                    prediction = self.predict(image_path)
                    
                    # Visualize comparison
                    label_path = matches[image_path_str]
                    output_path = output_dir / f"comparison_{Path(image_path).stem}.png"
                    self.visualize_comparison(prediction, label_path, str(output_path))
                    
                    # Save prediction as GeoTIFF
                    output_tiff = output_dir / f"pred_{Path(image_path).name}"
                    with rasterio.open(label_path) as src:
                        meta = src.meta.copy()
                        meta.update({'count': 2, 'height': 11, 'width': 11})
                        with rasterio.open(output_tiff, 'w', **meta) as dst:
                            dst.write(prediction)
                except Exception as e:
                    print(f"Error processing {image_path}: {str(e)}")
                    continue

def main():
    # Setup paths and create predictor
    checkpoint_path = "/mnt/e/soil-moisture/soil-moisture/checkpoints/rcnn_best.pth"
    predictor = SoilMoisturePredictor(checkpoint_path)
    
    # Setup paths
    image_folder = Path("/mnt/e/soil-moisture/soil-moisture/data/combine_tiffs_val")
    label_folder = "/mnt/e/soil-moisture/soil-moisture/data/label_tiffs_val"
    output_dir = "predictions"
    
    # Get all image paths
    image_paths = list(image_folder.glob("*.tif"))
    print(f"Found {len(image_paths)} total images")
    
    # Predict and compare with cleaned data
    try:
        predictor.predict_and_compare(image_paths, label_folder, output_dir)
    except Exception as e:
        print(f"Error during prediction: {str(e)}")

if __name__ == "__main__":
    main()
