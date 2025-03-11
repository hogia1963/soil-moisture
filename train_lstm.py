import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
import rasterio
from models.lstm_model import SoilMoistureLSTM
import joblib
from tqdm import tqdm
from typing import List, Tuple
from utils.data_cleaner import get_clean_data_pairs
from utils.data_organizer import DataOrganizer

class SoilMoistureSequenceDataset(Dataset):
    def __init__(self, sequences: List[List[Tuple[str, str]]], sequence_length: int = 5):
        self.sequences = sequences
        self.sequence_length = sequence_length
        
        # Load scalers
        self.image_scalers = [
            joblib.load(f'checkpoints/image_scaler_band_{i}.joblib')
            for i in range(21)
        ]
        self.label_scalers = [
            joblib.load(f'checkpoints/label_scaler_band_{i}.joblib')
            for i in range(2)
        ]
    
    def __len__(self):
        return len(self.sequences)
    
    def _transform_data(self, data: np.ndarray, scalers: List):
        transformed = np.zeros_like(data)
        for band in range(data.shape[0]):
            band_data = data[band]
            band_data = np.nan_to_num(band_data, nan=scalers[band].mean_[0])
            reshaped = band_data.reshape(-1, 1)
            transformed[band] = scalers[band].transform(reshaped).reshape(band_data.shape)
        return transformed
    
    def __getitem__(self, idx):
        # Get the sequence for this index
        sequence = self.sequences[idx]
        
        # Load images
        image_sequence = []
        for img_path, _ in sequence:
            with rasterio.open(img_path) as src:
                img = src.read().astype(np.float32)
                img = self._transform_data(img, self.image_scalers)
                image_sequence.append(img)
        
        # Load final label
        _, label_path = sequence[-1]
        with rasterio.open(label_path) as src:
            label = src.read().astype(np.float32)
            label = self._transform_data(label, self.label_scalers)
        
        return (torch.FloatTensor(np.stack(image_sequence)), 
                torch.FloatTensor(label))

def train_lstm_model(
    image_folder: str,
    label_folder: str,
    val_image_folder: str,
    val_label_folder: str,
    num_epochs: int = 200,
    batch_size: int = 8,
    sequence_length: int = 5,
    learning_rate: float = 0.001
) -> Tuple[dict, str]:
    
    # Get cleaned data pairs
    train_matches = get_clean_data_pairs(image_folder, label_folder)
    val_matches = get_clean_data_pairs(val_image_folder, val_label_folder)
    
    # Organize data by location
    train_organizer = DataOrganizer(train_matches)
    val_organizer = DataOrganizer(val_matches)
    
    # Get all sequences for all locations
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
    
    # Create datasets and dataloaders
    train_dataset = SoilMoistureSequenceDataset(train_sequences, sequence_length)
    val_dataset = SoilMoistureSequenceDataset(val_sequences, sequence_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SoilMoistureLSTM(
        input_channels=21,
        hidden_size=128,
        num_layers=2,
        sequence_length=sequence_length
    ).to(device)
    
    # Loss and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    best_val_loss = float('inf')
    best_model_path = None
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for sequences, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences = sequences.to(device)
                labels = labels.to(device)
                outputs = model(sequences)
                val_loss += criterion(outputs, labels).item()
        
        val_loss /= len(val_loader)
        avg_loss = total_loss / len(train_loader)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {avg_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = f'checkpoints/lstm_best.pth'
            torch.save(model.state_dict(), best_model_path)
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_loss,
                'val_loss': val_loss
            }, f'checkpoints/lstm_epoch_{epoch+1}.pth')
    
    return {'train_loss': avg_loss, 'val_loss': val_loss}, best_model_path

if __name__ == "__main__":
    history, best_model = train_lstm_model(
        image_folder="data/combine_tiffs",
        label_folder="data/label_tiffs",
        val_image_folder="data/combine_tiffs_val",
        val_label_folder="data/label_tiffs_val",
    )
    print(f"Training completed! Best model saved at {best_model}")
