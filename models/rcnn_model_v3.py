import torch
import torch.nn as nn
from .rcnn_model_v2 import ResBlock, CBAM  # Reuse components from v2

class RCNN_v3(nn.Module):
    def __init__(self, image_channels=21, history_channels=2):
        super(RCNN_v3, self).__init__()
        
        # Shared image encoder for both current and previous images
        self.image_encoder = nn.ModuleList([
            nn.Sequential(
                ResBlock(image_channels, 64, stride=2),
                CBAM(64)
            ),
            nn.Sequential(
                ResBlock(64, 128, stride=2),
                ResBlock(128, 128),
                CBAM(128)
            ),
            nn.Sequential(
                ResBlock(128, 256, stride=2),
                ResBlock(256, 256),
                CBAM(256)
            )
        ])
        
        # Fusion for current and previous image features
        self.temporal_fusion = nn.Sequential(
            ResBlock(512, 256),  # 256*2 -> 256
            CBAM(256)
        )
        
        # Decoder path
        self.decoder = nn.ModuleList([
            nn.Sequential(
                ResBlock(256, 128),
                nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
                CBAM(128)
            ),
            nn.Sequential(
                ResBlock(128, 64),
                nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
                CBAM(64)
            ),
            nn.Sequential(
                ResBlock(64, 32),
                nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
                CBAM(32)
            )
        ])
        
        # Light historical data processing
        self.history_processor = nn.Sequential(
            nn.Conv2d(history_channels, history_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(history_channels),
            nn.ReLU(inplace=True)
        )
        
        # Final fusion and output
        self.final = nn.Sequential(
            nn.AdaptiveAvgPool2d((11, 11)),
            nn.Conv2d(32 + history_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, kernel_size=1)
        )
        
        self.initialize_weights()
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def encode_image(self, x):
        """Shared image encoding path"""
        features = []
        for encoder in self.image_encoder:
            x = encoder(x)
            features.append(x)
        return x, features
    
    def forward(self, current_image, previous_image, history):
        # Handle NaN values
        current_image = torch.nan_to_num(current_image, nan=0.0)
        previous_image = torch.nan_to_num(previous_image, nan=0.0)
        history = torch.nan_to_num(history, nan=0.0)
        
        # Encode both current and previous images using shared encoder
        current_features, _ = self.encode_image(current_image)
        previous_features, _ = self.encode_image(previous_image)
        
        # Combine temporal image features
        combined_features = torch.cat([current_features, previous_features], dim=1)
        fused_features = self.temporal_fusion(combined_features)
        
        # Decoder path
        x = fused_features
        for decoder in self.decoder:
            x = decoder(x)
        
        # Process historical data
        x_hist = self.history_processor(history)
        x_hist = nn.functional.interpolate(
            x_hist, 
            size=x.shape[-2:],
            mode='bilinear',
            align_corners=False
        )
        
        # Final fusion and output
        x = torch.cat([x, x_hist], dim=1)
        output = self.final(x)
        
        return output
