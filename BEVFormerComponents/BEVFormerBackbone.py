'''
In this script, we'll be building a BEVFormer backbone using a pretrained ResNet50 + FPN backbone from torchvision.
'''

# Import modules we need
from torchvision.models import resnet50
from torchvision.ops import FeaturePyramidNetwork
import torch
import torch.nn as nn

class ResNetFPNBackbone(nn.Module):
    def __init__(self, pretrained = True, fpn_out_channels = 256):
        super().__init__()
        # Load a pretrained ResNet50 model
        resnet = resnet50(pretrained=pretrained)
        # Extract layers to use as backbone (Only use layers up to conv4)
        # Stem (first few conv layers)
        # Input:  [B, 3, 256, 704]
        # Output after stem: [B, 64, 64, 176]  Channels = 64 because Resnet's stem has 64 channels
        #   conv1: stride 2  → [B, 64, 128, 352]   Kernel size = 7, stride cuts h w dimensions in half
        #   maxpool: stride 2 → [B, 64, 64, 176]   Stride cuts the h w in half again
        #  normalize activations with bn1
        #  Nonlinearity with relu
        self.stem = torch.nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )
        # ResNet residual layers
        # layer1: stride 1  → [B, 256, 64, 176]  (C2)
        # layer2: stride 2  → [B, 512, 32, 88]   (C3)
        # layer3: stride 2  → [B, 1024, 16, 44]  (C4)
        # layer4: stride 2  → [B, 2048, 8, 22]   (C5)
        self.layer1 = resnet.layer1  # Output stride 4
        self.layer2 = resnet.layer2  # Output stride 8
        self.layer3 = resnet.layer3  # Output stride 16
        self.layer4 = resnet.layer4  # Output stride 32
       
        # FPN: unify all feature levels to 256 channels each
        #   Input channel list corresponds to C2–C5 above
        #   Outputs: each map has 256 channels, but different spatial sizes
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[256, 512, 1024, 2048],
            out_channels=fpn_out_channels,
        )
    
    # Forward Pass
    def forward(self, x):
        # Pass input through ResNet layers:
        # Each layer downsamples the spatial resolution (H, W) but increases the semantic richness (channels)
        # C2–C5 are the standard ResNet feature maps.
        # So, early layers (like the stem and C2) pick up fine-grained features — edges, corners, color gradients, tiny textures.
        # Later layers (C4, C5) pick up bigger patterns — wheels, pedestrians, road segments, vehicles — the “semantics.”
        x = self.stem(x)
        c2 = self.layer1(x)  # C2
        c3 = self.layer2(c2) # C3
        c4 = self.layer3(c3) # C4
        c5 = self.layer4(c4) # C5

        # Create a dict of feature maps for FPN
        features = {
            "C2": c2,
            "C3": c3,
            "C4": c4,
            "C5": c5
        }
        # Pass through FPN:
        # The FPN takes these different-resolution maps and produces a unified multi-scale feature hierarchy —
        # each at the same number of channels (e.g. 256), but preserving the multi-scale nature.
        # FPN upsamples the lower-resolution (coarse) feature maps step by step until they match the higher-resolution maps above them.
        # Top-down pathway
        #    P5 = conv1x1(C5)   // 8×22
        #    P4 = conv1x1(C4) + upsample(P5)    //16×44
        #    P3 = conv1x1(C3) + upsample(P4)    //32×88
        #    P2 = conv1x1(C2) + upsample(P3)    //64×176
        # Feature Pyramid Network (FPN) combines the features: it fuses both fine and coarse features, 
        # so the network can “see” both where things are (precise detail) and what they are (semantic meaning).
        fpn_features = self.fpn(features)
        # This is a dictionary of feature maps. Each map is a 2D grid of learned features, but at a different spatial scale:
        # {
        #    "C2": tensor([B, 256, 64, 176]),   # P2 fine edges, small objects
        #    "C3": tensor([B, 256, 32, 88]),    # P3 small/medium objects
        #    "C4": tensor([B, 256, 16, 44]),    # P4 medium/large objects
        #    "C5": tensor([B, 256, 8, 22])      # P5 large objects, global context
        #  }
        return fpn_features

if __name__ == "__main__":
    model = ResNetFPNBackbone(pretrained=True)
    dummy = torch.randn(1, 3, 256, 704)  # example image
    outs = model(dummy)
    for k, v in outs.items():
        print(k, v.shape)

