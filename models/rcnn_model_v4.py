import torch
import torch.nn as nn
from .rcnn_model_v2 import ResBlock, CBAM

class RCNN_v4(nn.Module):
    def __init__(self, image_channels=21, history_channels=2):
        super(RCNN_v4, self).__init__()
        
        # Image encoder (shared between t1 and t2)
        self.image_encoder = nn.ModuleList([
            nn.Sequential(
                ResBlock(image_channels, 64, stride=2),
                CBAM(64)
            ),
            nn.Sequential(
                ResBlock(64, 128, stride=2),
                CBAM(128)
            ),
            nn.Sequential(
                ResBlock(128, 256, stride=2),
                CBAM(256)
            )
        ])
        
        # Historical label encoder
        self.label_encoder = nn.Sequential(
            ResBlock(history_channels, 32),
            CBAM(32),
            ResBlock(32, 64),
            CBAM(64)
        )
        
        # Fusion module
        self.fusion = nn.Sequential(
            ResBlock(256 * 2 + 64, 256),  # Combine t1, t2 images and label features
            CBAM(256),
            ResBlock(256, 128)
        )
        
        # Modified decoder for better small difference handling
        self.decoder = nn.Sequential(
            ResBlock(128, 64),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ResBlock(64, 32),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ResBlock(32, 16),
            nn.AdaptiveAvgPool2d((11, 11)),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, history_channels, kernel_size=1),
            nn.Tanh()  # Add Tanh for bounded output
        )
        
        self.initialize_weights()
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def encode_image(self, x):
        features = []
        for encoder in self.image_encoder:
            x = encoder(x)
            features.append(x)
        return x, features
    
    def forward(self, image_t1, image_t2, label_t1):
        # Handle NaN values
        image_t1 = torch.nan_to_num(image_t1, nan=0.0)
        image_t2 = torch.nan_to_num(image_t2, nan=0.0)
        label_t1 = torch.nan_to_num(label_t1, nan=0.0)
        
        # Encode both images using shared encoder
        feat_t1, _ = self.encode_image(image_t1)
        feat_t2, _ = self.encode_image(image_t2)
        
        # Encode historical label
        label_feat = self.label_encoder(label_t1)
        
        # Resize label features if needed
        if label_feat.size(-1) != feat_t1.size(-1):
            label_feat = nn.functional.interpolate(
                label_feat,
                size=(feat_t1.size(-2), feat_t1.size(-1)),
                mode='bilinear',
                align_corners=True
            )
        
        # Fuse features
        fused = torch.cat([feat_t1, feat_t2, label_feat], dim=1)
        fused = self.fusion(fused)
        
        # Decode to get label difference
        diff_pred = self.decoder(fused)
        
        return diff_pred
