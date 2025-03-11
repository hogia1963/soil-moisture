import torch
import torch.nn as nn

class SoilMoistureLSTM(nn.Module):
    def __init__(self, input_channels=21, hidden_size=128, num_layers=2, sequence_length=5):
        super(SoilMoistureLSTM, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        
        # CNN feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.AdaptiveAvgPool2d((8, 8))  # Force 8x8 spatial dimensions
        )
        
        # Calculate flattened feature size
        self.feature_size = 128 * 8 * 8
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.feature_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        # Decoder to get back to spatial dimensions
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, self.feature_size),
            nn.ReLU(),
            nn.Linear(self.feature_size, 11 * 11 * 2),  # Output: 2 channels at 11x11
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Reshape input to process each timestep through CNN
        # From [batch, seq_len, channels, H, W] to [batch*seq_len, channels, H, W]
        x = x.view(-1, self.input_channels, x.size(-2), x.size(-1))
        
        # Extract features
        features = self.feature_extractor(x)
        
        # Reshape for LSTM: [batch, seq_len, feature_size]
        features = features.view(batch_size, self.sequence_length, -1)
        
        # Process through LSTM
        lstm_out, _ = self.lstm(features)
        
        # Take only the last timestep's output
        last_output = lstm_out[:, -1, :]
        
        # Decode to final output
        output = self.decoder(last_output)
        
        # Reshape to [batch, 2, 11, 11]
        return output.view(batch_size, 2, 11, 11)
