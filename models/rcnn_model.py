import torch
import torch.nn as nn

class RCNN(nn.Module):
    def __init__(self, input_channels=21):
        super(RCNN, self).__init__()
        
        def init_weights(m):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Calculate exact padding and kernel sizes to get 11x11 output
        self.encoder = nn.Sequential(
            # 222x222 -> 111x111
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # 111x111 -> 56x56
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # 56x56 -> 28x28
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # 28x28 -> 14x14
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        
        # Final processing to get exactly 11x11
        self.final = nn.Sequential(
            # 14x14 -> 11x11 using a specific kernel and stride
            nn.Conv2d(256, 128, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # Keep 11x11 resolution
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Final 2-channel output at 11x11 with tanh activation
            nn.Conv2d(64, 2, kernel_size=1),
            nn.Tanh()  # Add this line to constrain output to [-1, 1]
        )
        
        # Initialize weights
        self.encoder.apply(init_weights)
        self.final.apply(init_weights)
        
    def forward(self, x):
        # Add input validation
        if torch.any(torch.isnan(x)):
            print("NaN found in model input")
            x = torch.nan_to_num(x, nan=0.0)
            
        x = self.encoder(x)
        x = self.final(x)
        return x

    def debug_sizes(self, x):
        """Helper method to debug tensor sizes"""
        print(f"Input size: {x.size()}")
        
        # Encoder path
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                print(f"After encoder conv {i}: {x.size()}")
                
        # Final processing
        for i, layer in enumerate(self.final):
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                print(f"After final conv {i}: {x.size()}")
        
        return x
