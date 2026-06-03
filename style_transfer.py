#!/usr/bin/env python3
"""
Neural Style Transfer — Apply artistic styles to images.
Supports Gatys optimization and AdaIN feed-forward methods.
"""
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
from models.adain import AdaINTransfer
from models.gatys import GatysTransfer

def load_image(path: str, size: int = 512) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(path).convert("RGB")
    return transform(image).unsqueeze(0)

def save_image(tensor: torch.Tensor, path: str):
    """Save tensor as image."""
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    image = inv_normalize(tensor.squeeze(0)).clamp(0, 1)
    image = transforms.ToPILImage()(image)
    image.save(path)

def main():
    parser = argparse.ArgumentParser(description="Neural Style Transfer")
    parser.add_argument("--content", required=True, help="Content image path")
    parser.add_argument("--style", required=True, help="Style image path")
    parser.add_argument("--output", default="output.jpg", help="Output path")
    parser.add_argument("--method", choices=["gatys", "adain"], default="adain")
    parser.add_argument("--size", type=int, default=512, help="Output size")
    parser.add_argument("--steps", type=int, default=300, help="Optimization steps (Gatys)")
    parser.add_argument("--alpha", type=float, default=1.0, help="Style strength (AdaIN)")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    content = load_image(args.content, args.size).to(device)
    style = load_image(args.style, args.size).to(device)
    
    if args.method == "adain":
        print("Using AdaIN method...")
        model = AdaINTransfer().to(device)
        output = model.transfer(content, style, alpha=args.alpha)
    else:
        print(f"Using Gatys method ({args.steps} steps)...")
        model = GatysTransfer().to(device)
        output = model.transfer(content, style, steps=args.steps)
    
    save_image(output.cpu(), args.output)
    print(f"\n✅ Saved to {args.output}")

if __name__ == "__main__":
    main()
