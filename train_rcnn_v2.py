import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
import rasterio
from models.rcnn_model_v2 import RCNN_v2
from sklearn.preprocessing import StandardScaler
from utils.data_cleaner import get_clean_data_pairs
from tqdm import tqdm
import gc
from typing import Optional, Dict, List, Tuple
import joblib
import torch.nn as nn
from utils.data_organizer import DataOrganizer

class SoilMoistureDatasetV2(Dataset):
    def __init__(self, sequences: List[List[Tuple[str, str]]], force_refit=False):
        Path("checkpoints").mkdir(parents=True, exist_ok=True)
        
        self.sequences = sequences
        
        # Initialize scalers
        self.image_scalers = self._get_or_fit_scalers([seq[-1][0] for seq in sequences], 21, 'image', force_refit)
        self.label_scalers = self._get_or_fit_scalers([seq[-1][1] for seq in sequences], 2, 'label', force_refit)
        
    def _get_or_fit_scalers(self, file_paths, num_bands, prefix, force_refit=False):
        """Load existing scalers or fit new ones"""
        scalers = []
        all_scalers_exist = True
        
        # Check if all scaler files exist
        for band in range(num_bands):
            scaler_path = Path(f'checkpoints/{prefix}_scaler_band_{band}.joblib')
            if not scaler_path.exists():
                all_scalers_exist = False
                break
        
        # Load existing scalers if they exist and force_refit is False
        if all_scalers_exist and not force_refit:
            print(f"\nLoading existing {prefix} scalers...")
            for band in range(num_bands):
                scaler_path = f'checkpoints/{prefix}_scaler_band_{band}.joblib'
                scaler = joblib.load(scaler_path)
                scalers.append(scaler)
            return scalers
        
        # If not all scalers exist or force_refit is True, fit new ones
        print(f"\nFitting new {prefix} scalers...")
        return self._fit_scalers(file_paths, num_bands, prefix)

    def _fit_scalers(self, file_paths, num_bands, prefix):
        """Fit StandardScaler for each band"""
        scalers = [StandardScaler() for _ in range(num_bands)]
        
        for file_path in tqdm(file_paths, desc=f"Processing {prefix} files"):
            with rasterio.open(file_path) as src:
                data = src.read().astype(np.float32)
                
                for band in range(num_bands):
                    band_data = data[band]
                    valid_mask = ~np.isnan(band_data)
                    valid_data = band_data[valid_mask].reshape(-1, 1)
                    if len(valid_data) > 0:
                        scalers[band] = scalers[band].partial_fit(valid_data)
        
        # Save scalers
        for band, scaler in enumerate(scalers):
            joblib.dump(scaler, f'checkpoints/{prefix}_scaler_band_{band}.joblib')
        
        return scalers
    
    def _transform_data(self, data, scalers):
        """Transform each band using its corresponding scaler"""
        transformed = np.zeros_like(data)
        for band in range(data.shape[0]):
            band_data = data[band]
            band_data = np.nan_to_num(band_data, nan=scalers[band].mean_[0])
            reshaped = band_data.reshape(-1, 1)
            transformed[band] = scalers[band].transform(reshaped).reshape(band_data.shape)
        return transformed
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        # Get the sequence for this location
        sequence = self.sequences[idx]
        
        # Use current and previous images
        current_img_path, current_label_path = sequence[-1]  # Latest image
        prev_img_path, prev_label_path = sequence[-2]  # Previous image
        
        # Load and transform current image
        with rasterio.open(current_img_path) as src:
            current_image = src.read().astype(np.float32)
            current_image = self._transform_data(current_image, self.image_scalers)
        
        # Load and transform historical label
        with rasterio.open(prev_label_path) as src:
            historical_label = src.read().astype(np.float32)
            historical_label = self._transform_data(historical_label, self.label_scalers)
        
        # Load and transform current label (target)
        with rasterio.open(current_label_path) as src:
            current_label = src.read().astype(np.float32)
            current_label = self._transform_data(current_label, self.label_scalers)
        
        return (torch.FloatTensor(current_image), 
                torch.FloatTensor(historical_label),
                torch.FloatTensor(current_label))

def print_scaler_statistics(scalers, prefix):
    """Print statistics for each band's scaler"""
    print(f"\n{prefix} Scaling Statistics:")
    print("-" * 50)
    print(f"{'Band':>4} | {'Mean':>10} | {'Std':>10} | {'Min':>10} | {'Max':>10}")
    print("-" * 50)
    
    for i, scaler in enumerate(scalers):
        mean = scaler.mean_[0]
        std = scaler.scale_[0]
        # Calculate min and max from training data
        min_val = mean - 2 * std  # Approximate min
        max_val = mean + 2 * std  # Approximate max
        print(f"{i:4d} | {mean:10.4f} | {std:10.4f} | {min_val:10.4f} | {max_val:10.4f}")

class WeightedMSELoss(nn.Module):
    def __init__(self, power=2.0):
        """
        Custom loss that penalizes larger deviations more heavily
        Args:
            power: Power to raise the difference to (default=2.0 for quadratic growth)
        """
        super(WeightedMSELoss, self).__init__()
        self.power = power
        
    def forward(self, pred, target):
        # Calculate absolute difference
        diff = torch.abs(pred - target)
        
        # Apply higher penalty for larger differences
        weighted_diff = torch.pow(diff, self.power)
        
        # Mean over all dimensions
        loss = torch.mean(weighted_diff)
        return loss

def train_model_v2(
    image_folder: str,
    label_folder: str,
    val_image_folder: str,
    val_label_folder: str,
    num_epochs=400,
    batch_size=10,
    sequence_length=2,  # Minimum 2 for current and previous
    learning_rate=0.001,
    checkpoint_path: Optional[str] = None,
    start_epoch: int = 0,
    force_refit_scalers=False,
    model_class=None
) -> Tuple[Dict, str]:
    
    # Get cleaned data pairs
    train_matches = get_clean_data_pairs(image_folder, label_folder)
    val_matches = get_clean_data_pairs(val_image_folder, val_label_folder)
    
    # Organize data by location
    train_organizer = DataOrganizer(train_matches)
    val_organizer = DataOrganizer(val_matches)
    
    # Get sequences for all locations
    train_sequences = []
    for location_sequences in train_organizer.get_all_sequences(sequence_length).values():
        train_sequences.extend(location_sequences)
    
    val_sequences = []
    for location_sequences in val_organizer.get_all_sequences(sequence_length).values():
        val_sequences.extend(location_sequences)
    
    # Print data organization statistics
    print("\nTraining Data:")
    train_organizer.print_statistics()
    print("\nValidation Data:")
    val_organizer.print_statistics()
    
    # Create datasets
    train_dataset = SoilMoistureDatasetV2(train_sequences, force_refit=force_refit_scalers)
    val_dataset = SoilMoistureDatasetV2(val_sequences, force_refit=False)
    
    # Print scaling statistics
    print("\nData Scaling Information:")
    print("=" * 60)
    print_scaler_statistics(train_dataset.image_scalers, "Image")
    print("\n")
    print_scaler_statistics(train_dataset.label_scalers, "Label")
    print("=" * 60)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Initialize model with specified class or default to RCNN_v2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ModelClass = model_class if model_class is not None else RCNN_v2
    model = ModelClass(image_channels=21, history_channels=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Use custom loss instead of standard MSE
    criterion = WeightedMSELoss(power=2.5).to(device)  # Increase power for stronger penalty
    
    # Training loop
    best_val_loss = float('inf')
    best_model_path = None
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (images, historical_labels, current_labels) in enumerate(train_dataloader):
            images = images.to(device)
            historical_labels = historical_labels.to(device)
            current_labels = current_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images, historical_labels)
            loss = criterion(outputs, current_labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], "
                      f"Batch [{batch_idx+1}/{len(train_dataloader)}], "
                      f"Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_dataloader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, historical_labels, current_labels in val_dataloader:
                images = images.to(device)
                historical_labels = historical_labels.to(device)
                current_labels = current_labels.to(device)
                
                outputs = model(images, historical_labels)
                loss = criterion(outputs, current_labels)
                val_loss += loss.item()
        
        val_loss = val_loss / len(val_dataloader)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Training Loss: {avg_loss:.4f}, "
              f"Validation Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = f'checkpoints/rcnn_v2_best.pth'
            torch.save(model.state_dict(), best_model_path)
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f'checkpoints/rcnn_v2_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'val_loss': val_loss
            }, checkpoint_path)
    
    return {'train_loss': avg_loss, 'val_loss': val_loss}, best_model_path

if __name__ == "__main__":
    history, best_model = train_model_v2(
        image_folder="data/combine_tiffs",
        label_folder="data/label_tiffs",
        val_image_folder="data/combine_tiffs_val",
        val_label_folder="data/label_tiffs_val",
        checkpoint_path=None,
        start_epoch=0,
        force_refit_scalers=False
    )

    print(f"Training completed! Best model saved at {best_model}")
