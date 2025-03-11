import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
import rasterio
from models.rcnn_model_v4 import RCNN_v4
from utils.data_cleaner import get_clean_data_pairs
from utils.data_organizer import DataOrganizer
import joblib
from typing import List, Tuple, Dict
from tqdm import tqdm
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import sys

class SoilMoistureDiffDataset(Dataset):
    def __init__(self, sequences: List[List[Tuple[str, str]]], force_refit=False):
        self.sequences = sequences
        
        # Load scalers
        self.image_scalers = [
            joblib.load(f'checkpoints/image_scaler_band_{i}.joblib')
            for i in range(21)
        ]
        self.label_scalers = [
            joblib.load(f'checkpoints/label_scaler_band_{i}.joblib')
            for i in range(2)
        ]
        
        # Add difference statistics
        self.diff_mean = 0
        self.diff_std = 1
        if not force_refit:
            try:
                stats = np.load('checkpoints/label_diff_stats.npz')
                self.diff_mean = stats['mean']
                self.diff_std = stats['std']
                print(f"Loaded difference statistics - Mean: {self.diff_mean:.6f}, Std: {self.diff_std:.6f}")
            except:
                print("Computing difference statistics...")
                self.compute_diff_statistics()
    
    def compute_diff_statistics(self):
        """Compute mean and std of label differences"""
        all_diffs = []
        
        for idx in tqdm(range(len(self)), desc="Computing difference statistics"):
            seq_t1 = self.sequences[idx]
            seq_t2 = self.sequences[idx + 1]
            
            # Load labels
            with rasterio.open(seq_t1[-1][1]) as src:
                label_t1 = self._transform_data(src.read().astype(np.float32), self.label_scalers)
            with rasterio.open(seq_t2[-1][1]) as src:
                label_t2 = self._transform_data(src.read().astype(np.float32), self.label_scalers)
            
            diff = label_t2 - label_t1
            all_diffs.append(diff)
        
        all_diffs = np.concatenate([d.reshape(-1) for d in all_diffs])
        self.diff_mean = float(np.mean(all_diffs))
        self.diff_std = float(np.std(all_diffs))
        
        # Save statistics
        np.savez('checkpoints/label_diff_stats.npz', 
                 mean=self.diff_mean, 
                 std=self.diff_std)
        
        print(f"Difference statistics - Mean: {self.diff_mean:.6f}, Std: {self.diff_std:.6f}")
    
    def normalize_diff(self, diff):
        """Normalize difference using computed statistics"""
        return (diff - self.diff_mean) / (self.diff_std + 1e-6)
    
    def denormalize_diff(self, norm_diff):
        """Convert normalized difference back to original scale"""
        return norm_diff * self.diff_std + self.diff_mean
    
    def __len__(self):
        return len(self.sequences) - 1  # Need pairs of consecutive timestamps
    
    def _transform_data(self, data: np.ndarray, scalers: List) -> np.ndarray:
        transformed = np.zeros_like(data)
        for band in range(data.shape[0]):
            band_data = data[band]
            band_data = np.nan_to_num(band_data, nan=scalers[band].mean_[0])
            reshaped = band_data.reshape(-1, 1)
            transformed[band] = scalers[band].transform(reshaped).reshape(band_data.shape)
        return transformed
    
    def __getitem__(self, idx):
        # Get consecutive sequences
        seq_t1 = self.sequences[idx]
        seq_t2 = self.sequences[idx + 1]
        
        # Load data for t1
        img_path_t1, label_path_t1 = seq_t1[-1]
        with rasterio.open(img_path_t1) as src:
            image_t1 = self._transform_data(src.read().astype(np.float32), self.image_scalers)
        with rasterio.open(label_path_t1) as src:
            label_t1 = self._transform_data(src.read().astype(np.float32), self.label_scalers)
            
        # Load data for t2
        img_path_t2, label_path_t2 = seq_t2[-1]
        with rasterio.open(img_path_t2) as src:
            image_t2 = self._transform_data(src.read().astype(np.float32), self.image_scalers)
        with rasterio.open(label_path_t2) as src:
            label_t2 = self._transform_data(src.read().astype(np.float32), self.label_scalers)
        
        # Calculate and normalize label difference
        label_diff = label_t2 - label_t1
        norm_label_diff = self.normalize_diff(label_diff)
        
        return (torch.FloatTensor(image_t1),
                torch.FloatTensor(image_t2),
                torch.FloatTensor(label_t1),
                torch.FloatTensor(norm_label_diff))

def visualize_training(history: Dict[str, List[float]], save_path: str = "training_plots"):
    """Visualize training and validation losses"""
    save_dir = Path(save_path)
    save_dir.mkdir(exist_ok=True)
    
    # Plot losses
    plt.figure(figsize=(12, 6))
    plt.plot(history['train_losses'], label='Training Loss', alpha=0.8)
    plt.plot(history['val_losses'], label='Validation Loss', alpha=0.8)
    plt.title('Training and Validation Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_dir / 'loss_history.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot difference statistics
    if 'avg_pred_diff' in history and 'avg_true_diff' in history:
        plt.figure(figsize=(12, 6))
        plt.plot(history['avg_pred_diff'], label='Predicted Diff', alpha=0.8)
        plt.plot(history['avg_true_diff'], label='True Diff', alpha=0.8)
        plt.title('Average Predicted vs True Differences')
        plt.xlabel('Step')
        plt.ylabel('Difference')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(save_dir / 'diff_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

def print_metrics(metrics: Dict[str, float], step: int, phase: str = "train"):
    """Print formatted metrics"""
    header = f"[{phase.upper()} Step {step}]"
    metrics_str = " | ".join([f"{k}: {v:.6f}" for k, v in metrics.items()])
    print(f"{header} {metrics_str}")

def calculate_metrics(pred_diff: np.ndarray, true_diff: np.ndarray) -> Dict[str, float]:
    """Calculate detailed metrics for difference prediction"""
    metrics = {}
    
    # Handle batch dimension
    # pred_diff and true_diff shape: [batch_size, channels, height, width]
    
    # Mean metrics across all dimensions
    metrics['mean_abs_error'] = float(np.mean(np.abs(pred_diff - true_diff)))
    metrics['mean_error'] = float(np.mean(pred_diff - true_diff))
    
    # Per-channel metrics
    for i in range(pred_diff.shape[1]):  # Iterate over channels dimension
        # Calculate metrics for each channel across batch dimension
        channel_pred = pred_diff[:, i]  # Shape: [batch_size, height, width]
        channel_true = true_diff[:, i]
        
        metrics[f'channel_{i}_mse'] = float(np.mean((channel_pred - channel_true)**2))
        metrics[f'channel_{i}_mae'] = float(np.mean(np.abs(channel_pred - channel_true)))
        metrics[f'channel_{i}_mean'] = float(np.mean(channel_pred - channel_true))
    
    return metrics

def print_metrics_inline(metrics: Dict[str, float], epoch: int, batch_idx: int, num_batches: int):
    """Print metrics on the same line, updating in place"""
    metrics_str = " | ".join([f"{k}: {v:.6f}" for k, v in metrics.items()])
    status = f"\rEpoch [{epoch}] Batch [{batch_idx}/{num_batches}] {metrics_str}"
    sys.stdout.write(status)
    sys.stdout.flush()

def train_model_v4(
    image_folder: str,
    label_folder: str,
    val_image_folder: str,
    val_label_folder: str,
    num_epochs: int = 200,
    batch_size: int = 8,
    sequence_length: int = 2,
    learning_rate: float = 0.001,
    checkpoint_path: str = None  # Add checkpoint path parameter
) -> Tuple[Dict, str]:
    
    # Get cleaned data pairs and organize by location
    train_matches = get_clean_data_pairs(image_folder, label_folder)
    val_matches = get_clean_data_pairs(val_image_folder, val_label_folder)
    
    train_organizer = DataOrganizer(train_matches)
    val_organizer = DataOrganizer(val_matches)
    
    # Get sequences
    train_sequences = []
    for location_sequences in train_organizer.get_all_sequences(sequence_length).values():
        train_sequences.extend(location_sequences)
    
    val_sequences = []
    for location_sequences in val_organizer.get_all_sequences(sequence_length).values():
        val_sequences.extend(location_sequences)
    
    # Create datasets and dataloaders
    train_dataset = SoilMoistureDiffDataset(train_sequences)
    val_dataset = SoilMoistureDiffDataset(val_sequences)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RCNN_v4(image_channels=21, history_channels=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()  # Add loss criterion
    
    # Enhanced history tracking
    history = {
        'train_losses': [],
        'val_losses': [],
        'avg_pred_diff': [],
        'avg_true_diff': [],
        'learning_rates': [],
        'train_metrics': [],
        'val_metrics': []
    }
    
    # Load checkpoint if provided
    start_epoch = 0
    best_val_loss = float('inf')
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"\nLoading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        # Load training history if available
        if 'history' in checkpoint:
            history = checkpoint['history']
            print(f"Continuing from epoch {start_epoch} with best validation loss: {best_val_loss:.6f}")
    
    print("\nStarting training from epoch", start_epoch)
    print("=" * 80)
    
    # Training loop with progress bars
    best_val_loss = float('inf')
    best_model_path = None
    step = 0
    
    epoch_pbar = tqdm(range(num_epochs), desc="Training Progress", position=0)
    
    for epoch in epoch_pbar:
        model.train()
        epoch_metrics = {
            'loss': 0,
            'mean_abs_error': 0,
            'channel_0_mse': 0,
            'channel_1_mse': 0
        }
        
        # Progress bar for batches
        batch_pbar = tqdm(enumerate(train_loader), 
                         total=len(train_loader),
                         desc=f"Epoch {epoch+1}",
                         leave=False,
                         position=1)
        
        for batch_idx, batch in batch_pbar:
            image_t1, image_t2, label_t1, norm_label_diff = [b.to(device) for b in batch]
            
            optimizer.zero_grad()
            pred_norm_diff = model(image_t1, image_t2, label_t1)
            loss = criterion(pred_norm_diff, norm_label_diff)
            loss.backward()
            optimizer.step()
            
            # Calculate and update metrics
            with torch.no_grad():
                pred_diff = train_dataset.denormalize_diff(pred_norm_diff.detach().cpu().numpy())
                true_diff = train_dataset.denormalize_diff(norm_label_diff.detach().cpu().numpy())
                
                metrics = calculate_metrics(pred_diff, true_diff)
                metrics['loss'] = loss.item()
                
                # Update progress bar description with current metrics
                batch_pbar.set_postfix({k: f"{v:.4f}" for k, v in metrics.items()})
                
                # Update history
                history['train_losses'].append(loss.item())
                history['train_metrics'].append(metrics)
                
                # Update epoch metrics
                for k, v in metrics.items():
                    epoch_metrics[k] = epoch_metrics.get(k, 0) + v
        
        # Validation phase with progress bar
        model.eval()
        val_metrics = {k: 0 for k in epoch_metrics.keys()}
        
        val_pbar = tqdm(val_loader, 
                       desc="Validation",
                       leave=False,
                       position=1)
        
        with torch.no_grad():
            for batch in val_pbar:
                image_t1, image_t2, label_t1, norm_label_diff = [b.to(device) for b in batch]
                pred_norm_diff = model(image_t1, image_t2, label_t1)
                loss = criterion(pred_norm_diff, norm_label_diff)
                
                # Calculate validation metrics
                pred_diff = train_dataset.denormalize_diff(pred_norm_diff.cpu().numpy())
                true_diff = train_dataset.denormalize_diff(norm_label_diff.cpu().numpy())
                batch_metrics = calculate_metrics(pred_diff, true_diff)
                batch_metrics['loss'] = loss.item()
                
                # Update validation metrics
                for k, v in batch_metrics.items():
                    val_metrics[k] += v
                
                # Update progress bar
                val_pbar.set_postfix({k: f"{v/len(val_loader):.4f}" for k, v in val_metrics.items()})
        
        # Average validation metrics
        val_metrics = {k: v/len(val_loader) for k, v in val_metrics.items()}
        history['val_metrics'].append(val_metrics)
        
        # Update epoch progress bar with summary metrics
        epoch_metrics = {k: v/len(train_loader) for k, v in epoch_metrics.items()}
        summary_metrics = {
            'train_loss': epoch_metrics['loss'],
            'val_loss': val_metrics['loss']
        }
        epoch_pbar.set_postfix(summary_metrics)
        
        # Save best model if needed
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_model_path = f'checkpoints/rcnn_v4_best.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
                'val_loss': val_metrics['loss']
            }, best_model_path)
            
            visualize_training(history)
        
        # Save regular checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
                'train_loss': epoch_metrics['loss'],
                'val_loss': val_metrics['loss']
            }, f'checkpoints/rcnn_v4_epoch_{epoch+1}.pth')
    
    return history, best_model_path

if __name__ == "__main__":
    # Example usage with checkpoint
    checkpoint_path = "/mnt/e/soil-moisture/soil-moisture/checkpoints/rcnn_v4_best.pth"  # Change this to your checkpoint path
    
    history, best_model = train_model_v4(
        image_folder="data/combine_tiffs",
        label_folder="data/label_tiffs",
        val_image_folder="data/combine_tiffs_val",
        val_label_folder="data/label_tiffs_val",
        checkpoint_path=checkpoint_path  # Add checkpoint path
    )
    print(f"Training completed! Best model saved at {best_model}")
