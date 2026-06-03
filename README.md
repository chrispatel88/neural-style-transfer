# 🎨 Neural Style Transfer

Neural style transfer engine for images and videos — combine content images with artistic styles using deep learning, optimized for AMD ROCm GPU acceleration.

## Features

- **Image Style Transfer**: Apply artistic styles to photos
- **Video Style Transfer**: Frame-by-frame video stylization with temporal consistency
- **Multiple Methods**: Gatys optimization, AdaIN, WCT, StyleSwap
- **Pre-trained Models**: VGG-19, VGG-16, ResNet feature extractors
- **Batch Processing**: Process multiple images/videos in parallel
- **Real-time Preview**: Live preview during optimization
- **ROCm Optimized**: HIP-optimized kernels for AMD GPUs

## Style Transfer Methods

| Method | Speed | Quality | GPU Memory |
|--------|-------|---------|------------|
| Gatys (optimization) | Slow | Highest | 4GB |
| AdaIN (feed-forward) | Fast | Good | 2GB |
| WCT | Medium | High | 6GB |
| StyleSwap | Fast | Good | 3GB |

## Quick Start

```bash
pip install -r requirements.txt

# Transfer style from painting to photo
python style_transfer.py \
  --content photos/city.jpg \
  --style paintings/starry_night.jpg \
  --output output/city_starry.jpg

# Video style transfer
python video_transfer.py \
  --input videos/input.mp4 \
  --style paintings/mondrian.jpg \
  --output output/stylized.mp4

# Batch process directory
python batch_transfer.py \
  --input-dir photos/ \
  --style paintings/wave.jpg \
  --output-dir output/ \
  --method adain
```

## Gallery

```
Content Image          Style Image            Output
┌─────────────┐       ┌─────────────┐       ┌─────────────┐
│             │       │             │       │             │
│   Photo     │   +   │  Painting   │   =   │  Stylized   │
│             │       │             │       │             │
└─────────────┘       └─────────────┘       └─────────────┘
```

## Architecture

```
Input Image → VGG Encoder → Style Features ─┐
                                             ├─ AdaIN → Decoder → Output
Style Image → VGG Encoder → Content Features─┘
```

## License

MIT
