import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = self.relu(out)
        return out

class CBAM(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction_ratio, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction_ratio, channels, 1)
        )
        # Spatial Attention
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.BatchNorm2d(1)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Channel Attention
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_out = self.sigmoid(avg_out + max_out)
        x = x * channel_out
        
        # Spatial Attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_in = torch.cat([avg_out, max_out], dim=1)
        spatial_out = self.sigmoid(self.conv(spatial_in))
        x = x * spatial_out
        return x

class FeatureFusion(nn.Module):
    def __init__(self, img_channels, hist_channels, output_channels):
        super(FeatureFusion, self).__init__()
        self.conv_img = nn.Conv2d(img_channels, output_channels, kernel_size=1)
        self.conv_hist = nn.Conv2d(hist_channels, output_channels, kernel_size=1)
        self.fusion_conv = nn.Conv2d(output_channels * 2, output_channels, kernel_size=1)
        
    def forward(self, img_feat, hist_feat):
        img_feat = self.conv_img(img_feat)
        hist_feat = self.conv_hist(hist_feat)
        
        if hist_feat.size(-1) != img_feat.size(-1):
            hist_feat = nn.functional.interpolate(
                hist_feat, 
                size=(img_feat.size(-2), img_feat.size(-1)),
                mode='bilinear',
                align_corners=False
            )
        
        combined = torch.cat([img_feat, hist_feat], dim=1)
        fused = self.fusion_conv(combined)
        return fused

class RCNN_v2(nn.Module):
    def __init__(self, image_channels=21, history_channels=2):
        super(RCNN_v2, self).__init__()
        
        # Image encoder path
        self.image_encoder = nn.ModuleList([
            nn.Sequential(
                ResBlock(image_channels, 64, stride=2),  # 222->111
                CBAM(64)
            ),
            nn.Sequential(
                ResBlock(64, 128, stride=2),  # 111->56
                ResBlock(128, 128),
                CBAM(128)
            ),
            nn.Sequential(
                ResBlock(128, 256, stride=2),  # 56->28
                ResBlock(256, 256),
                CBAM(256)
            )
        ])
        
        # Decoder path for image features
        self.decoder = nn.ModuleList([
            nn.Sequential(
                ResBlock(256, 128),
                nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),  # 28->56
                CBAM(128)
            ),
            nn.Sequential(
                ResBlock(128, 64),
                nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),  # 56->111
                CBAM(64)
            ),
            nn.Sequential(
                ResBlock(64, 32),
                nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),  # 111->222
                CBAM(32)
            )
        ])
        
        # Minimal historical data processing (preserve spatial dimensions)
        self.history_processor = nn.Sequential(
            nn.Conv2d(history_channels, history_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(history_channels),
            nn.ReLU(inplace=True)
        )
        
        # Final layers with size control
        self.final = nn.Sequential(
            nn.AdaptiveAvgPool2d((11, 11)),  # Both streams to 11x11
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
    
    def forward(self, image, history):
        # Handle NaN values
        image = torch.nan_to_num(image, nan=0.0)
        history = torch.nan_to_num(history, nan=0.0)
        
        # Process image through encoder-decoder
        x_img = image
        for encoder in self.image_encoder:
            x_img = encoder(x_img)
        
        for decoder in self.decoder:
            x_img = decoder(x_img)
        
        # Minimal processing for historical data
        x_hist = self.history_processor(history)
        
        # Resize historical data to match image features
        x_hist = nn.functional.interpolate(
            x_hist, 
            size=x_img.shape[-2:],
            mode='bilinear',
            align_corners=False
        )
        
        # Debug prints
        # print(f"Image features shape: {x_img.shape}")
        # print(f"Historical features shape: {x_hist.shape}")
        
        # Combine features
        x = torch.cat([x_img, x_hist], dim=1)
        
        # Final processing with size control
        output = self.final(x)
        # print(f"Output shape: {output.shape}")
        
        # Verify output size
        assert output.shape[-2:] == (11, 11), f"Expected output size (11,11), got {output.shape[-2:]}"
        return output
