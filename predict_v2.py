import torch
import rasterio
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib.colors import CenteredNorm
from predict import SoilMoisturePredictor
from basemodel import SoilModel
from tqdm import tqdm
from typing import Union, List, Dict, Tuple
import joblib
from utils.soil_apis import get_soil_data
from utils.inference_class import SoilMoistureInferencePreprocessor
from match_images_labels import match_images_with_labels

class ModelComparator:
    def __init__(self, rcnn_checkpoint: str, base_checkpoint: str, device: str = None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize both models
        self.rcnn_predictor = SoilMoisturePredictor(rcnn_checkpoint)
        self.base_model = SoilModel.load_from_checkpoint(base_checkpoint).to(self.device)
        self.base_model.eval()
        
        # Initialize preprocessor for base model
        self.preprocessor = SoilMoistureInferencePreprocessor()
        
    def predict_both(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Get predictions from both models"""
        # RCNN prediction
        rcnn_pred = self.rcnn_predictor.predict(image_path)
        
        # Base model prediction
        with torch.no_grad():
            base_inputs = self.preprocessor.preprocess(image_path)
            for key in base_inputs:
                if isinstance(base_inputs[key], torch.Tensor):
                    base_inputs[key] = base_inputs[key].to(self.device)
            base_outputs = self.base_model.run_inference(base_inputs)
            base_pred = np.stack([base_outputs['surface'], base_outputs['rootzone']])
            
        return rcnn_pred, base_pred
    
    def visualize_comparison(self, rcnn_pred: np.ndarray, base_pred: np.ndarray, 
                           ground_truth: np.ndarray, save_path: str = None):
        """Visualize predictions from both models compared to ground truth"""
        # Handle NaN values in all arrays
        ground_truth = np.nan_to_num(ground_truth, nan=0.0)
        rcnn_pred = np.nan_to_num(rcnn_pred, nan=0.0)
        base_pred = np.nan_to_num(base_pred, nan=0.0)
        
        # Create masks for valid data (non-NaN in original ground truth)
        valid_mask_surface = ~np.isnan(ground_truth[0])
        valid_mask_subsurface = ~np.isnan(ground_truth[1])
        
        # Calculate metrics only on valid data
        surface_mse_rcnn = mean_squared_error(
            ground_truth[0][valid_mask_surface],
            rcnn_pred[0][valid_mask_surface]
        )
        surface_mse_base = mean_squared_error(
            ground_truth[0][valid_mask_surface],
            base_pred[0][valid_mask_surface]
        )
        subsurface_mse_rcnn = mean_squared_error(
            ground_truth[1][valid_mask_subsurface],
            rcnn_pred[1][valid_mask_subsurface]
        )
        subsurface_mse_base = mean_squared_error(
            ground_truth[1][valid_mask_subsurface],
            base_pred[1][valid_mask_subsurface]
        )
        
        # Create visualization
        fig = plt.figure(figsize=(25, 10))
        gs = gridspec.GridSpec(2, 5, figure=fig)
        
        # Surface row visualization (top row)
        vmin_surface = min(np.nanmin(rcnn_pred[0]), np.nanmin(base_pred[0]), np.nanmin(ground_truth[0]))
        vmax_surface = max(np.nanmax(rcnn_pred[0]), np.nanmax(base_pred[0]), np.nanmax(ground_truth[0]))
        
        # Plot surface predictions and differences
        titles_surface = ['RCNN Surface', 'Base Surface', 'Ground Truth Surface']
        data_surface = [rcnn_pred[0], base_pred[0], ground_truth[0]]
        
        for i, (title, img) in enumerate(zip(titles_surface, data_surface)):
            ax = fig.add_subplot(gs[0, i])
            im = ax.imshow(img, cmap='viridis', vmin=vmin_surface, vmax=vmax_surface)
            ax.set_title(title)
            plt.colorbar(im, ax=ax)
        
        # Surface differences
        diff_surface_rcnn = np.where(valid_mask_surface, rcnn_pred[0] - ground_truth[0], np.nan)
        diff_surface_base = np.where(valid_mask_surface, base_pred[0] - ground_truth[0], np.nan)
        max_abs_diff_surface = max(
            np.nanmax(np.abs(diff_surface_rcnn)),
            np.nanmax(np.abs(diff_surface_base))
        )
        norm_surface = CenteredNorm(vcenter=0, halfrange=max_abs_diff_surface)
        
        # Plot surface differences
        ax = fig.add_subplot(gs[0, 3])
        im = ax.imshow(diff_surface_rcnn, cmap='RdBu_r', norm=norm_surface)
        ax.set_title('Surface: RCNN vs GT')
        plt.colorbar(im, ax=ax)
        
        ax = fig.add_subplot(gs[0, 4])
        im = ax.imshow(diff_surface_base, cmap='RdBu_r', norm=norm_surface)
        ax.set_title('Surface: Base vs GT')
        plt.colorbar(im, ax=ax)
        
        # Subsurface moisture comparison (bottom row)
        vmin_subsurface = min(np.nanmin(rcnn_pred[1]), np.nanmin(base_pred[1]), np.nanmin(ground_truth[1]))
        vmax_subsurface = max(np.nanmax(rcnn_pred[1]), np.nanmax(base_pred[1]), np.nanmax(ground_truth[1]))
        
        titles_subsurface = ['RCNN Subsurface', 'Base Subsurface', 'Ground Truth Subsurface',
                            'RCNN vs GT Diff', 'Base vs GT Diff']
        data_subsurface = [rcnn_pred[1], base_pred[1], ground_truth[1]]
        
        # First three columns: predictions
        for i, (title, img) in enumerate(zip(titles_subsurface[:3], data_subsurface)):
            ax = fig.add_subplot(gs[1, i])
            im = ax.imshow(img, cmap='viridis', vmin=vmin_subsurface, vmax=vmax_subsurface)
            ax.set_title(title)
            plt.colorbar(im, ax=ax)
        
        # Last two columns: differences
        diff_subsurface_rcnn = rcnn_pred[1] - ground_truth[1]
        diff_subsurface_base = base_pred[1] - ground_truth[1]
        max_abs_diff_subsurface = max(
            np.abs(diff_subsurface_rcnn).max(),
            np.abs(diff_subsurface_base).max()
        )
        norm_subsurface = CenteredNorm(vcenter=0, halfrange=max_abs_diff_subsurface)
        
        # RCNN difference
        ax = fig.add_subplot(gs[1, 3])
        im = ax.imshow(diff_subsurface_rcnn, cmap='RdBu_r', norm=norm_subsurface)
        ax.set_title('Subsurface: RCNN vs GT')
        plt.colorbar(im, ax=ax)
        
        # Base difference
        ax = fig.add_subplot(gs[1, 4])
        im = ax.imshow(diff_subsurface_base, cmap='RdBu_r', norm=norm_subsurface)
        ax.set_title('Subsurface: Base vs GT')
        plt.colorbar(im, ax=ax)
        
        # Add metrics text
        surface_mse_rcnn = mean_squared_error(ground_truth[0], rcnn_pred[0])
        surface_mse_base = mean_squared_error(ground_truth[0], base_pred[0])
        subsurface_mse_rcnn = mean_squared_error(ground_truth[1], rcnn_pred[1])
        subsurface_mse_base = mean_squared_error(ground_truth[1], base_pred[1])
        
        plt.figtext(0.02, 0.95, f'Surface MSE - RCNN: {surface_mse_rcnn:.4f}, Base: {surface_mse_base:.4f}')
        plt.figtext(0.02, 0.45, f'Subsurface MSE - RCNN: {subsurface_mse_rcnn:.4f}, Base: {subsurface_mse_base:.4f}')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    
    def _calculate_metrics(self, ground_truth, prediction, valid_mask):
        """Calculate metrics safely handling empty or invalid data"""
        if not valid_mask.any():  # If mask is all False
            return None
            
        gt_valid = ground_truth[valid_mask]
        pred_valid = prediction[valid_mask]
        
        if len(gt_valid) == 0 or len(pred_valid) == 0:
            return None
            
        try:
            return mean_squared_error(gt_valid, pred_valid)
        except Exception as e:
            print(f"Error calculating MSE: {str(e)}")
            return None
    
    def compare_models(self, image_paths: List[str], label_folder: str, output_dir: str = "comparisons"):
        """Compare predictions from both models across multiple images"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Match images with labels using existing function
        matches = match_images_with_labels(str(Path(image_paths[0]).parent), label_folder)
        
        results = {
            'rcnn': {'surface_mse': [], 'subsurface_mse': []},
            'base': {'surface_mse': [], 'subsurface_mse': []}
        }
        
        for image_path in tqdm(image_paths, desc="Comparing models"):
            image_path_str = str(image_path)
            if image_path_str in matches:
                try:
                    # Get predictions from both models
                    rcnn_pred, base_pred = self.predict_both(image_path)
                    
                    # Load ground truth using matched label path
                    label_path = matches[image_path_str]
                    with rasterio.open(label_path) as src:
                        ground_truth = src.read().astype(np.float32)
                    
                    # Create masks for valid data
                    valid_mask_surface = ~np.isnan(ground_truth[0])
                    valid_mask_subsurface = ~np.isnan(ground_truth[1])
                    
                    # Calculate metrics only if we have valid data
                    surface_mse_rcnn = self._calculate_metrics(
                        ground_truth[0], rcnn_pred[0], valid_mask_surface
                    )
                    if surface_mse_rcnn is not None:
                        results['rcnn']['surface_mse'].append(surface_mse_rcnn)
                    
                    subsurface_mse_rcnn = self._calculate_metrics(
                        ground_truth[1], rcnn_pred[1], valid_mask_subsurface
                    )
                    if subsurface_mse_rcnn is not None:
                        results['rcnn']['subsurface_mse'].append(subsurface_mse_rcnn)
                    
                    surface_mse_base = self._calculate_metrics(
                        ground_truth[0], base_pred[0], valid_mask_surface
                    )
                    if surface_mse_base is not None:
                        results['base']['surface_mse'].append(surface_mse_base)
                    
                    subsurface_mse_base = self._calculate_metrics(
                        ground_truth[1], base_pred[1], valid_mask_subsurface
                    )
                    if subsurface_mse_base is not None:
                        results['base']['subsurface_mse'].append(subsurface_mse_base)
                    
                    # Save comparison visualization
                    output_path = output_dir / f"comparison_{Path(image_path).stem}.png"
                    self.visualize_comparison(rcnn_pred, base_pred, ground_truth, str(output_path))
                
                except Exception as e:
                    print(f"Error processing {image_path}: {str(e)}")
                    continue
            else:
                print(f"No matching label found for {image_path}")
        
        # Print metrics only if we have valid results
        if any(len(results[model][metric]) > 0 
               for model in results 
               for metric in results[model]):
            print("\nAverage Metrics:")
            for model in ['rcnn', 'base']:
                for metric in ['surface_mse', 'subsurface_mse']:
                    if results[model][metric]:
                        avg = np.mean(results[model][metric])
                        print(f"{model.upper()} {metric}: {avg:.4f}")
                    else:
                        print(f"{model.upper()} {metric}: No valid data")
        else:
            print("No valid comparisons were made!")

def main():
    rcnn_checkpoint = "/mnt/e/soil-moisture/soil-moisture/checkpoints/rcnn_epoch_200.pth"
    base_checkpoint = "tasks/defined_tasks/soilmoisture/SoilModel.ckpt"
    
    comparator = ModelComparator(rcnn_checkpoint, base_checkpoint)
    
    # Setup paths
    image_folder = Path("/mnt/e/soil-moisture/soil-moisture/data/combine_tiffs_val")
    label_folder = "/mnt/e/soil-moisture/soil-moisture/data/label_tiffs_validation"
    output_dir = "model_comparisons"
    
    # Compare models
    image_paths = list(image_folder.glob("*.tif"))
    comparator.compare_models(image_paths, label_folder, output_dir)

if __name__ == "__main__":
    main()
