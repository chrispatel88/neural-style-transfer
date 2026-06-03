"""Gatys optimization-based style transfer."""
import torch
import torch.nn as nn
import torchvision.models as models

class GatysTransfer(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(pretrained=True).features
        self.slices = nn.ModuleList([
            nn.Sequential(*vgg[:2]),   # relu1_1
            nn.Sequential(*vgg[2:7]),  # relu2_1
            nn.Sequential(*vgg[7:12]), # relu3_1
            nn.Sequential(*vgg[12:21]), # relu4_1
        ])
        
        for param in self.parameters():
            param.requires_grad = False
    
    def gram_matrix(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        features = x.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (c * h * w)
    
    def get_features(self, x: torch.Tensor) -> list:
        features = []
        h = x
        for slice in self.slices:
            h = slice(h)
            features.append(h)
        return features
    
    def transfer(self, content: torch.Tensor, style: torch.Tensor, 
                 steps: int = 300, content_weight: float = 1.0, 
                 style_weight: float = 1e6) -> torch.Tensor:
        """Optimize image to match content and style."""
        target = content.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([target], lr=0.01)
        
        content_feats = self.get_features(content)
        style_feats = self.get_features(style)
        style_grams = [self.gram_matrix(f) for f in style_feats]
        
        for step in range(steps):
            optimizer.zero_grad()
            
            target_feats = self.get_features(target)
            
            # Content loss
            content_loss = nn.functional.mse_loss(target_feats[-1], content_feats[-1])
            
            # Style loss
            style_loss = 0
            for tf, sg in zip(target_feats, style_grams):
                style_loss += nn.functional.mse_loss(self.gram_matrix(tf), sg)
            
            total_loss = content_weight * content_loss + style_weight * style_loss
            total_loss.backward()
            optimizer.step()
            
            if (step + 1) % 50 == 0:
                print(f"  Step {step+1}/{steps} | Content: {content_loss:.4f} | Style: {style_loss:.4f}")
        
        return target.detach()
