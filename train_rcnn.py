import torch
import torch.nn as nn  # Add this import
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
from models.rcnn_model import RCNN
from match_images_labels import match_images_with_labels
import rasterio
import gc
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler  # Add this import
import joblib
from typing import Optional, Dict, Tuple
from utils.data_cleaner import get_clean_data_pairs
from utils.losses import CombinedLoss  # Add this import

class SoilMoistureDataset(Dataset):
    def __init__(self, image_paths, label_paths, force_refit=False):
        # Create checkpoints directory if it doesn't exist
        Path("checkpoints").mkdir(parents=True, exist_ok=True)
        
        self.image_paths = image_paths
        self.label_paths = label_paths
        
        # Initialize scalers with option to load existing ones
        self.image_scalers = self._get_or_fit_scalers(image_paths, num_bands=21, prefix='image', force_refit=force_refit)
        self.label_scalers = self._get_or_fit_scalers(label_paths, num_bands=2, prefix='label', force_refit=force_refit)
    
    def _get_or_fit_scalers(self, file_paths, num_bands, prefix, force_refit=False):
        """Load existing scalers or fit new ones if they don't exist"""
        scalers = []
        all_scalers_exist = True
        
        # Check if all scaler files exist with new minmax suffix
        for band in range(num_bands):
            scaler_path = Path(f'checkpoints/{prefix}_scaler_band_{band}_minmax.joblib')
            if not scaler_path.exists():
                all_scalers_exist = False
                break
        
        # Load existing scalers if they exist and force_refit is False
        if all_scalers_exist and not force_refit:
            print(f"\nLoading existing {prefix} MinMax scalers...")
            for band in range(num_bands):
                scaler_path = f'checkpoints/{prefix}_scaler_band_{band}_minmax.joblib'
                scaler = joblib.load(scaler_path)
                scalers.append(scaler)
                print(f"{prefix} Band {band} stats - min: {scaler.data_min_[0]:.4f}, max: {scaler.data_max_[0]:.4f}")
            return scalers
        
        # If not all scalers exist or force_refit is True, fit new ones
        print(f"\nFitting new {prefix} scalers...")
        return self._fit_scalers(file_paths, num_bands, prefix)

    def _fit_scalers(self, file_paths, num_bands, prefix):
        """Fit MinMaxScaler for each band"""
        print(f"\nFitting scalers for {prefix} data across {len(file_paths)} files...")
        
        # Initialize scalers for each band with range [-1, 1]
        scalers = [MinMaxScaler(feature_range=(-1, 1)) for _ in range(num_bands)]
        
        # First pass: collect data for fitting
        for file_path in tqdm(file_paths, desc=f"Processing {prefix} files"):
            with rasterio.open(file_path) as src:
                data = src.read().astype(np.float32)
                
                # Process each band
                for band in range(num_bands):
                    band_data = data[band]
                    valid_mask = ~np.isnan(band_data) & ~np.isinf(band_data)
                    valid_data = band_data[valid_mask].reshape(-1, 1)
                    if len(valid_data) > 0:
                        scalers[band] = scalers[band].partial_fit(valid_data)
        
        # Save scalers with new minmax suffix
        for band, scaler in enumerate(scalers):
            joblib.dump(scaler, f'checkpoints/{prefix}_scaler_band_{band}_minmax.joblib')
            print(f"{prefix} Band {band} stats - min: {scaler.data_min_[0]:.4f}, max: {scaler.data_max_[0]:.4f}")
        
        return scalers
    
    def _transform_data(self, data, scalers):
        """Transform each band using its corresponding scaler"""
        transformed = np.zeros_like(data)
        for band in range(data.shape[0]):
            band_data = data[band]
            # Replace inf/nan with min of the band
            band_data = np.nan_to_num(band_data, nan=scalers[band].data_min_[0])
            # Reshape for scaler
            reshaped = band_data.reshape(-1, 1)
            # Transform and reshape back
            transformed[band] = scalers[band].transform(reshaped).reshape(band_data.shape)
        
        return transformed
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        # Load and transform image
        with rasterio.open(self.image_paths[idx]) as src:
            image = src.read().astype(np.float32)
            image = self._transform_data(image, self.image_scalers)
            
        # Load and transform label
        with rasterio.open(self.label_paths[idx]) as src:
            label = src.read().astype(np.float32)
            label = self._transform_data(label, self.label_scalers)
            # Ensure labels are in [-1, 1] range
            label = np.clip(label, -1, 1)
        
        return torch.FloatTensor(image), torch.FloatTensor(label)

def validate_model(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: str) -> Tuple[float, Dict]:
    """Validate model and return average validation loss and metrics"""
    model.eval()
    total_loss = 0
    metrics_sum = {'mse': 0, 'l1': 0, 'ssim': 0}
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            outputs = model(images)
            outputs = torch.clamp(outputs, -1, 1)
            loss_dict = criterion(outputs, labels)
            
            total_loss += loss_dict['total'].item()
            for key in metrics_sum:
                metrics_sum[key] += loss_dict[key]
            
            del outputs
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Average the metrics
    avg_metrics = {k: v / len(dataloader) for k, v in metrics_sum.items()}
    return total_loss / len(dataloader), avg_metrics  # Fixed return statement

def train_model(image_folder: str, label_folder: str, val_image_folder: str, val_label_folder: str, 
                num_epochs: int = 400, batch_size: int = 10, learning_rate: float = 0.001, 
                checkpoint_path: Optional[str] = None, start_epoch: int = 0, 
                force_refit_scalers: bool = False) -> Tuple[Dict, str]:
    
    # Create checkpoints directory if it doesn't exist
    Path("checkpoints").mkdir(parents=True, exist_ok=True)
    
    # Get matched and cleaned data pairs for training
    print("Matching and cleaning training data pairs...")
    train_matches = get_clean_data_pairs(
        image_folder, 
        label_folder,
        max_nan_percentage=50.0,
        save_report=True
    )
    
    # Get matched and cleaned data pairs for validation
    print("Matching and cleaning validation data pairs...")
    val_matches = get_clean_data_pairs(
        val_image_folder,
        val_label_folder,
        max_nan_percentage=50.0,
        save_report=True
    )
    
    if not train_matches or not val_matches:
        raise ValueError("No valid image-label pairs found after cleaning!")
    
    # Prepare training data
    train_image_paths = list(train_matches.keys())
    train_label_paths = list(train_matches.values())
    
    # Prepare validation data
    val_image_paths = list(val_matches.keys())
    val_label_paths = list(val_matches.values())
    
    print(f"Training with {len(train_matches)} clean image-label pairs")
    print(f"Validating with {len(val_matches)} clean image-label pairs")
    
    # Initialize datasets and dataloaders
    train_dataset = SoilMoistureDataset(train_image_paths, train_label_paths, force_refit=force_refit_scalers)
    val_dataset = SoilMoistureDataset(val_image_paths, val_label_paths, force_refit=False)
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Initialize model and training components
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RCNN(input_channels=21).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = CombinedLoss(mse_weight=1.0, l1_weight=0.5, ssim_weight=0.5)
    
    # Initialize best model tracking
    best_val_loss = float('inf')
    best_model_path = None
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'train_mse': [],
        'train_l1': [],
        'train_ssim': [],
        'val_mse': [],
        'val_l1': [],
        'val_ssim': [],
        'best_epoch': None
    }
    
    # Load checkpoint if provided
    if checkpoint_path:
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                if 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'best_val_loss' in checkpoint:
                    best_val_loss = checkpoint['best_val_loss']
                if 'training_history' in checkpoint:
                    training_history = checkpoint['training_history']
        else:
            model.load_state_dict(checkpoint)
    
    # Add gradient clipping
    max_grad_norm = 1.0
    
    # Training loop with progress bar
    epoch_pbar = tqdm(range(start_epoch, num_epochs), desc='Epochs', position=0)
    
    for epoch in epoch_pbar:
        # Training phase
        model.train()
        total_train_loss = 0
        train_metrics_sum = {'mse': 0, 'l1': 0, 'ssim': 0}
        
        batch_pbar = tqdm(enumerate(train_dataloader), 
                         total=len(train_dataloader),
                         desc=f'Epoch {epoch+1}/{num_epochs}',
                         position=1, 
                         leave=False)
        
        for batch_idx, (images, labels) in batch_pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Debug prints
            if torch.any(torch.isnan(images)) or torch.any(torch.isnan(labels)):
                print(f"NaN found in input batch {batch_idx}")
                continue
                
            outputs = model(images)
            # Clip outputs to [-1, 1] range
            outputs = torch.clamp(outputs, -1, 1)
            
            # Debug prints
            if torch.any(torch.isnan(outputs)):
                print(f"NaN found in model output batch {batch_idx}")
                continue
                
            # Calculate loss using combined criterion
            loss_dict = criterion(outputs, labels)
            loss = loss_dict['total']
            
            # Skip bad loss
            if torch.isnan(loss):
                print(f"NaN loss at batch {batch_idx}, skipping...")
                continue
                
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()
            
            # Update metrics
            total_train_loss += loss.item()
            for key in train_metrics_sum:
                train_metrics_sum[key] += loss_dict[key]
            
            # Print statistics periodically
            if batch_idx % 10 == 0:
                print(f"\nBatch {batch_idx} metrics:")
                print(f"Total loss: {loss.item():.4f}")
                for key, value in loss_dict.items():
                    if key != 'total':
                        print(f"{key}: {value:.4f}")
            
            # Update progress bars
            batch_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Clear memory
            del outputs
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Calculate average metrics
        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_train_metrics = {k: v / len(train_dataloader) for k, v in train_metrics_sum.items()}
        
        # Validation phase
        val_loss, val_metrics = validate_model(model, val_dataloader, criterion, device)
        
        # Update history
        training_history['train_loss'].append(avg_train_loss)
        training_history['val_loss'].append(val_loss)
        for key in ['mse', 'l1', 'ssim']:
            training_history[f'train_{key}'].append(avg_train_metrics[key])
            training_history[f'val_{key}'].append(val_metrics[key])
        
        # Update progress bar
        epoch_pbar.set_postfix({
            'train_loss': f'{avg_train_loss:.4f}',
            'val_loss': f'{val_loss:.4f}',
            'train_ssim': f'{avg_train_metrics["ssim"]:.4f}'
        })

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            training_history['best_epoch'] = epoch
            best_model_path = f'checkpoints/rcnn_best.pth'
            
            # Save best model checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'training_history': training_history
            }
            torch.save(checkpoint, best_model_path)
        
        # Save regular checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'training_history': training_history
            }
            torch.save(checkpoint, f'checkpoints/rcnn_epoch_{epoch+1}.pth')
    
    epoch_pbar.close()
    return training_history, best_model_path

if __name__ == "__main__":
    image_folder = "data/combine_tiffs"
    label_folder = "data/label_tiffs"
    val_image_folder = "data/combine_tiffs_val"
    val_label_folder = "data/label_tiffs_val"
    
    checkpoint_path = None
    start_epoch = 0
    
    history, best_model = train_model(
        image_folder=image_folder,
        label_folder=label_folder,
        val_image_folder=val_image_folder,
        val_label_folder=val_label_folder,
        checkpoint_path=checkpoint_path,
        start_epoch=start_epoch,
        force_refit_scalers=False
    )
    
    # Print final results
    print("\nTraining completed!")
    print(f"Best model saved at: {best_model}")
    print(f"Best validation loss: {min(history['val_loss']):.4f}")
    print(f"Best epoch: {history['best_epoch']}")
