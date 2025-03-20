# Real Estate Video Generator

Convert static real estate listings into engaging TikTok-style videos automatically.

## Features

- Transforms static property images into dynamic, cinematic videos
- Adds text overlays from property descriptions
- Applies professional transitions, ken burns effect, and motion
- Generates 16:9 aspect ratio videos perfect for TikTok, Instagram, and other platforms
- Customizable video duration (30-60 seconds)
- Background music options
- Adds text callouts for key property features

## Demo

[Example Video](https://github.com/quinnwins/real-estate-video-generator/assets/example.mp4)

## Installation

```bash
# Clone the repository
git clone https://github.com/quinnwins/real-estate-video-generator.git
cd real-estate-video-generator

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python generate_video.py --images_dir ./sample_data/images --description ./sample_data/description.txt --output video.mp4
```

### Advanced Options

```bash
python generate_video.py \
  --images_dir ./my_listing/photos \
  --description ./my_listing/desc.txt \
  --output my_listing_video.mp4 \
  --duration 45 \
  --music ambient1 \
  --text_style modern \
  --highlight_features \
  --add_logo ./my_logo.png
```

## Configuration

Edit `config.yaml` to customize default settings:

```yaml
# Example configuration
video:
  fps: 30
  resolution: "1920x1080"  # 16:9 aspect ratio
  min_duration_per_image: 2
  max_duration_per_image: 4

effects:
  zoom_range: [1.0, 1.2]
  pan_speed: 0.05
  transition_type: "dissolve"
  
text:
  font: "Montserrat"
  title_size: 48
  body_size: 28
  color: "#FFFFFF"
  shadow: True
```

## Requirements

- Python 3.8+
- MoviePy
- OpenCV
- Pillow
- PyYAML

## How It Works

1. **Image Processing**: The tool analyzes and prepares each image for video conversion
2. **Motion Generation**: Applies dynamic effects to static images using the Ken Burns effect
3. **Text Extraction**: Parses property description to extract key selling points
4. **Audio Integration**: Adds background music and optional voiceover
5. **Rendering**: Compiles everything into a polished, professional video

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
