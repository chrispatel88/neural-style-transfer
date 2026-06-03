"""AdaIN (Adaptive Instance Normalization) style transfer."""
import torch
import torch.nn as nn
import torchvision.models as models

class VGGEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(pretrained=True).features
        self.enc_1 = nn.Sequential(*vgg[:2])   # relu1_1
        self.enc_2 = nn.Sequential(*vgg[2:7])  # relu2_1
        self.enc_3 = nn.Sequential(*vgg[7:12]) # relu3_1
        self.enc_4 = nn.Sequential(*vgg[12:21]) # relu4_1
    
    def forward(self, x):
        h1 = self.enc_1(x)
        h2 = self.enc_2(h1)
        h3 = self.enc_3(h2)
        h4 = self.enc_4(h3)
        return h1, h2, h3, h4

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1),
        )
    
    def forward(self, x):
        return self.decoder(x)

class AdaINTransfer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = VGGEncoder()
        self.decoder = Decoder()
    
    def adain(self, content_feat: torch.Tensor, style_feat: torch.Tensor) -> torch.Tensor:
        """Adaptive Instance Normalization."""
        c_mean = content_feat.mean(dim=[2, 3], keepdim=True)
        c_std = content_feat.std(dim=[2, 3], keepdim=True) + 1e-6
        s_mean = style_feat.mean(dim=[2, 3], keepdim=True)
        s_std = style_feat.std(dim=[2, 3], keepdim=True) + 1e-6
        
        normalized = (content_feat - c_mean) / c_std
        return normalized * s_std + s_mean
    
    def transfer(self, content: torch.Tensor, style: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """Transfer style from style image to content image."""
        with torch.no_grad():
            c_feats = self.encoder(content)
            s_feats = self.encoder(style)
            
            # AdaIN on deepest features
            t = self.adain(c_feats[-1], s_feats[-1])
            t = alpha * t + (1 - alpha) * c_feats[-1]
            
            # Decode
            output = self.decoder(t)
        
        return output
