"""
U-Net for Segmentation - Solution

Complete U-Net implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two consecutive conv-bn-relu blocks."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """U-Net for image segmentation."""
    
    def __init__(self, in_channels: int = 1, out_channels: int = 2, features: list = None):
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]
        
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        
        # Encoder
        for feature in features:
            self.encoder.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        
        # Decoder
        for feature in reversed(features):
            self.decoder.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.decoder.append(DoubleConv(feature * 2, feature))
        
        # Final conv
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []
        
        # Encoder
        for encoder in self.encoder:
            x = encoder(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        skip_connections = skip_connections[::-1]
        
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)  # Upsample
            skip = skip_connections[idx // 2]
            
            # Handle size mismatch
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            
            x = torch.cat([skip, x], dim=1)  # Concat skip connection
            x = self.decoder[idx + 1](x)  # DoubleConv
        
        return self.final_conv(x)


class UNet3D(nn.Module):
    """3D U-Net for volumetric segmentation (CT, MRI)."""
    
    def __init__(self, in_channels: int = 1, out_channels: int = 2, features: list = None):
        super().__init__()
        if features is None:
            features = [32, 64, 128, 256]
        
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool3d(2, 2)
        
        for feature in features:
            self.encoder.append(self._double_conv_3d(in_channels, feature))
            in_channels = feature
        
        self.bottleneck = self._double_conv_3d(features[-1], features[-1] * 2)
        
        for feature in reversed(features):
            self.decoder.append(nn.ConvTranspose3d(feature * 2, feature, kernel_size=2, stride=2))
            self.decoder.append(self._double_conv_3d(feature * 2, feature))
        
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)
    
    def _double_conv_3d(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        skips = []
        for enc in self.encoder:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        skips = skips[::-1]
        
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)
            skip = skips[idx // 2]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=True)
            x = torch.cat([skip, x], dim=1)
            x = self.decoder[idx + 1](x)
        
        return self.final_conv(x)


class DiceLoss(nn.Module):
    """Dice loss for segmentation."""
    
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.softmax(pred, dim=1)
        
        if target.dim() == 3:  # (B, H, W) -> one-hot
            target = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
        
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


if __name__ == "__main__":
    torch.manual_seed(42)
    
    unet = UNet(in_channels=1, out_channels=2, features=[64, 128, 256, 512])
    
    x = torch.randn(2, 1, 256, 256)
    out = unet(x)
    
    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in unet.parameters()):,}")
    
    # Test 3D U-Net
    print("\n3D U-Net:")
    unet3d = UNet3D(in_channels=1, out_channels=3, features=[16, 32, 64])
    x3d = torch.randn(1, 1, 32, 64, 64)
    out3d = unet3d(x3d)
    print(f"Input: {x3d.shape}")
    print(f"Output: {out3d.shape}")
