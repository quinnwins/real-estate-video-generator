#!/usr/bin/env python3
"""
Real Estate Video Generator
---------------------------
This script takes real estate images and description text to create
an engaging TikTok-style video with dynamic effects.
"""

import os
import sys
import argparse
import yaml
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import cv2
from PIL import Image, ImageEnhance
from moviepy.editor import (
    ImageClip, TextClip, CompositeVideoClip, AudioFileClip,
    concatenate_videoclips, vfx, transfx
)
import moviepy.video.fx.all as vfx
from tqdm import tqdm
import nltk
from nltk.tokenize import sent_tokenize
import spacy
import glob
import time
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Download required NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Load spaCy model for text processing
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.error("Spacy model not found. Please install it with: python -m spacy download en_core_web_sm")
    sys.exit(1)

class RealEstateVideoGenerator:
    """Main class to generate real estate videos from static images and text."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the video generator with configuration.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.config = self._load_config(config_path)
        self.width, self.height = map(int, self.config['video']['resolution'].split('x'))
        self.fps = self.config['video']['fps']
        self.transitions = self.config['effects']['transitions']
        
        # Create asset directories if they don't exist
        os.makedirs("assets/music", exist_ok=True)
        os.makedirs("assets/branding", exist_ok=True)
        os.makedirs("output", exist_ok=True)
    
    def _load_config(self, config_path: str) -> Dict:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Returns:
            Dict containing configuration settings
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            sys.exit(1)
    
    def _load_images(self, images_dir: str) -> List[str]:
        """
        Load image paths from the specified directory.
        
        Args:
            images_dir: Directory containing real estate images
            
        Returns:
            List of image file paths
        """
        image_extensions = ['.jpg', '.jpeg', '.png', '.webp']
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(images_dir, f"*{ext}")))
        
        if not image_paths:
            logger.error(f"No images found in {images_dir}")
            sys.exit(1)
        
        # Sort images to ensure consistent ordering
        image_paths.sort()
        logger.info(f"Found {len(image_paths)} images")
        
        return image_paths
    
    def _extract_key_points(self, description: str) -> List[str]:
        """
        Extract key selling points from property description.
        
        Args:
            description: Property description text
            
        Returns:
            List of key points to highlight in the video
        """
        # Parse description with spaCy
        doc = nlp(description)
        
        # Break text into sentences
        sentences = [sent.text.strip() for sent in doc.sents]
        
        # Filter for sentences containing important features or highlight words
        key_points = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Check if sentence contains important features
            for feature in self.config['text_processing']['important_features']:
                if feature.lower() in sentence_lower:
                    key_points.append(sentence)
                    break
            
            # Check if sentence contains highlight words
            if sentence not in key_points:
                for word in self.config['text_processing']['highlight_words']:
                    if word.lower() in sentence_lower:
                        key_points.append(sentence)
                        break
        
        # Limit number of key points
        max_highlights = self.config['text_processing']['key_features_to_extract']
        if len(key_points) > max_highlights:
            key_points = key_points[:max_highlights]
        
        return key_points
    
    def _apply_ken_burns(self, img_path: str, duration: float) -> ImageClip:
        """
        Apply Ken Burns effect (pan and zoom) to an image.
        
        Args:
            img_path: Path to the image file
            duration: Duration of the clip in seconds
            
        Returns:
            ImageClip with Ken Burns effect applied
        """
        # Load and resize image to 16:9 if needed
        img = Image.open(img_path)
        img_width, img_height = img.size
        target_ratio = self.width / self.height
        
        # Check if we need to crop the image to match 16:9
        img_ratio = img_width / img_height
        
        if abs(img_ratio - target_ratio) > 0.01:  # If ratios don't match (with small tolerance)
            if img_ratio > target_ratio:
                # Image is wider than 16:9, crop width
                new_width = int(img_height * target_ratio)
                left = (img_width - new_width) // 2
                img = img.crop((left, 0, left + new_width, img_height))
            else:
                # Image is taller than 16:9, crop height
                new_height = int(img_width / target_ratio)
                top = (img_height - new_height) // 2
                img = img.crop((0, top, img_width, top + new_height))
        
        # Enhance image if specified in config
        if 'brightness_adjustment' in self.config['effects']:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(self.config['effects']['brightness_adjustment'])
        
        if 'contrast_adjustment' in self.config['effects']:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(self.config['effects']['contrast_adjustment'])
        
        if 'saturation_adjustment' in self.config['effects']:
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(self.config['effects']['saturation_adjustment'])
        
        # Resize the image to match video resolution
        img = img.resize((self.width, self.height), Image.LANCZOS)
        
        # Create the clip from the image
        clip = ImageClip(np.array(img)).set_duration(duration)
        
        # Randomly choose zoom start/end and direction
        zoom_range = self.config['effects']['zoom_range']
        zoom_start = random.uniform(zoom_range[0], zoom_range[1])
        zoom_end = random.uniform(zoom_range[0], zoom_range[1])
        
        # Apply Ken Burns effect with zooming and panning
        pan_range_x = self.config['effects']['pan_range_x']
        pan_range_y = self.config['effects']['pan_range_y']
        
        # Pick random pan directions
        pan_x_start = random.uniform(pan_range_x[0], pan_range_x[1])
        pan_x_end = random.uniform(pan_range_x[0], pan_range_x[1])
        pan_y_start = random.uniform(pan_range_y[0], pan_range_y[1])
        pan_y_end = random.uniform(pan_range_y[0], pan_range_y[1])

        def ken_burns_effect(t):
            # Calculate the progress (0 to 1)
            progress = t / duration
            
            # Interpolate zoom and pan values
            zoom = zoom_start + (zoom_end - zoom_start) * progress
            pan_x = pan_x_start + (pan_x_end - pan_x_start) * progress
            pan_y = pan_y_start + (pan_y_end - pan_y_start) * progress
            
            # Apply transformations
            zoomed = vfx.resize(lambda t: zoom)(clip).get_frame(t)
            
            # Calculate panning offsets
            h, w = zoomed.shape[:2]
            
            # Add extra width and height for panning
            pad_x = int(w * abs(pan_x))
            pad_y = int(h * abs(pan_y))
            
            # Coordinates to slice the image
            x1 = max(0, int(w * (pan_x + 0.5) - self.width // 2))
            y1 = max(0, int(h * (pan_y + 0.5) - self.height // 2))
            x2 = min(w, x1 + self.width)
            y2 = min(h, y1 + self.height)
            
            # Ensure we have the correct dimensions
            if x2 - x1 < self.width:
                x1 = max(0, w - self.width)
                x2 = w
            
            if y2 - y1 < self.height:
                y1 = max(0, h - self.height)
                y2 = h
            
            # Extract the visible portion
            frame = zoomed[y1:y2, x1:x2]
            
            # Resize to ensure we maintain the required dimensions
            if frame.shape[0] != self.height or frame.shape[1] != self.width:
                frame = cv2.resize(frame, (self.width, self.height))
            
            return frame
        
        # Return a clip applying the effect
        return ImageClip(ken_burns_effect, duration=duration).set_fps(self.fps)
    
    def _create_text_overlay(self, text: str, style: str = 'body', duration: float = None) -> TextClip:
        """
        Create a text overlay clip.
        
        Args:
            text: Text content
            style: Text style ('title', 'subtitle', or 'body')
            duration: Duration of the text clip
            
        Returns:
            TextClip object
        """
        if not duration:
            duration = self.config['text']['display_time']
            
        # Set text properties based on style
        if style == 'title':
            font = self.config['text']['title_font']
            fontsize = self.config['text']['title_size']
            color = self.config['text']['primary_color']
            position = self.config['text']['title_position']
        elif style == 'subtitle':
            font = self.config['text']['body_font']
            fontsize = self.config['text']['subtitle_size']
            color = self.config['text']['secondary_color']
            position = self.config['text']['title_position']
            # Adjust position slightly down for subtitle
            position = (position[0], position[1] + 0.08)
        else:  # body
            font = self.config['text']['body_font']
            fontsize = self.config['text']['body_size']
            color = self.config['text']['primary_color']
            position = self.config['text']['body_position']
        
        # Create text clip
        text_clip = TextClip(
            text, 
            fontsize=fontsize, 
            color=color,
            font=font,
            align='center',
            stroke_color=self.config['text']['shadow_color'],
            stroke_width=1
        )
        
        # Calculate position based on normalized coordinates
        pos_x = int(position[0] * self.width - text_clip.w / 2)
        pos_y = int(position[1] * self.height - text_clip.h / 2)
        
        # Add fade in/out effects
        fade_in_time = self.config['text']['fade_in_time']
        fade_out_time = self.config['text']['fade_out_time']
        
        text_clip = (text_clip
                    .set_position((pos_x, pos_y))
                    .set_duration(duration)
                    .fadein(fade_in_time)
                    .fadeout(fade_out_time))
        
        return text_clip
    
    def _add_branding(self, clip: CompositeVideoClip) -> CompositeVideoClip:
        """
        Add logo and branding to the video.
        
        Args:
            clip: Base video clip
            
        Returns:
            Video clip with branding elements added
        """
        logo_path = self.config['branding']['logo_path']
        
        # Check if logo file exists
        if not os.path.exists(logo_path):
            logger.warning(f"Logo file not found at {logo_path}. Skipping branding.")
            return clip
        
        # Load and resize logo
        logo_size = self.config['branding']['logo_size']
        logo_position = self.config['branding']['logo_position']
        logo_opacity = self.config['branding']['logo_opacity']
        
        logo_clip = (ImageClip(logo_path)
                    .resize(width=logo_size[0])
                    .set_opacity(logo_opacity)
                    .set_duration(clip.duration))
        
        # Calculate position based on normalized coordinates
        pos_x = int(logo_position[0] * self.width - logo_clip.w / 2)
        pos_y = int(logo_position[1] * self.height - logo_clip.h / 2)
        
        logo_clip = logo_clip.set_position((pos_x, pos_y))
        
        # Add logo to video
        return CompositeVideoClip([clip, logo_clip])
    
    def _add_music(self, clip: CompositeVideoClip, music_option: str = None) -> CompositeVideoClip:
        """
        Add background music to the video.
        
        Args:
            clip: Video clip
            music_option: Name of the music option to use
            
        Returns:
            Video clip with background music
        """
        # Get music options from config
        music_options = self.config['audio']['music_options']
        music_path = None
        
        # If a specific music option is requested, find it
        if music_option:
            for option in music_options:
                if option['name'] == music_option:
                    music_path = option['file']
                    break
        
        # If no matching option or none specified, choose randomly
        if not music_path and music_options:
            music_path = random.choice(music_options)['file']
        
        # Check if music file exists
        if not music_path or not os.path.exists(music_path):
            logger.warning(f"Music file not found at {music_path}. No music will be added.")
            return clip
        
        # Load audio and set volume
        audio_clip = (AudioFileClip(music_path)
                     .volumex(self.config['audio']['music_volume'])
                     .set_duration(clip.duration)
                     .audio_fadein(self.config['audio']['fade_in'])
                     .audio_fadeout(self.config['audio']['fade_out']))
        
        # Add audio to video
        return clip.set_audio(audio_clip)
    
    def generate_video(self, 
                       images_dir: str, 
                       description_file: str, 
                       output_path: str, 
                       duration: int = None,
                       music: str = None) -> str:
        """
        Generate a real estate video from images and description.
        
        Args:
            images_dir: Directory containing property images
            description_file: Path to text file with property description
            output_path: Path to save the output video
            duration: Desired video duration in seconds (overrides config)
            music: Music option to use (overrides random selection)
            
        Returns:
            Path to the generated video file
        """
        start_time = time.time()
        logger.info("Starting video generation process...")
        
        # Load images
        image_paths = self._load_images(images_dir)
        
        # Load description
        try:
            with open(description_file, 'r') as f:
                description = f.read()
        except Exception as e:
            logger.error(f"Error reading description file: {e}")
            sys.exit(1)
        
        # Extract key points from description
        key_points = self._extract_key_points(description)
        logger.info(f"Extracted {len(key_points)} key points from description")
        
        # Determine video duration
        if not duration:
            # Use random duration within the specified range
            duration_range = self.config['video']['duration_range']
            duration = random.randint(duration_range[0], duration_range[1])
        
        logger.info(f"Generating a {duration}-second video")
        
        # Calculate clips duration
        num_images = len(image_paths)
        transition_duration = self.config['video']['transition_duration']
        
        # Calculate total time needed for transitions
        total_transition_time = (num_images - 1) * transition_duration
        
        # Calculate available time for actual clips
        available_time = duration - total_transition_time
        
        # Calculate average time per image
        min_duration = self.config['video']['min_duration_per_image']
        max_duration = self.config['video']['max_duration_per_image']
        
        # Ensure we don't exceed min/max durations
        avg_duration = available_time / num_images
        clip_duration = max(min_duration, min(avg_duration, max_duration))
        
        # Adjust number of images if needed
        if avg_duration < min_duration:
            # Too many images for the duration, need to use fewer
            num_images = int(available_time / min_duration)
            image_paths = image_paths[:num_images]
            logger.warning(f"Too many images for the requested duration. Using only {num_images} images.")
        
        # Generate video clips with Ken Burns effect for each image
        logger.info("Generating clips with Ken Burns effect...")
        clips = []
        
        for i, img_path in enumerate(tqdm(image_paths)):
            # Apply Ken Burns effect
            clip = self._apply_ken_burns(img_path, clip_duration)
            
            # Add key point text if available
            if i < len(key_points) and self.config['text_processing']['highlight_important_features']:
                text_clip = self._create_text_overlay(key_points[i], 'body', clip_duration)
                clip = CompositeVideoClip([clip, text_clip])
            
            clips.append(clip)
        
        # Apply transitions between clips
        logger.info("Applying transitions between clips...")
        final_clips = []
        
        for i in range(len(clips)):
            clip = clips[i]
            
            # Skip transition for the last clip
            if i < len(clips) - 1:
                # Choose a random transition type
                transition_type = random.choice(self.transitions)
                
                if transition_type == "dissolve":
                    clip = clip.crossfadein(transition_duration)
                elif transition_type == "fade":
                    clip = clip.fadeout(transition_duration)
                elif transition_type == "wipe_left":
                    # Use predefined transition from MoviePy
                    clip = transfx.slide_out(clip, duration=transition_duration, side='left')
                elif transition_type == "wipe_right":
                    clip = transfx.slide_out(clip, duration=transition_duration, side='right')
                elif transition_type == "slide_left":
                    clip = transfx.slide_in(clip, duration=transition_duration, side='right')
                elif transition_type == "slide_right":
                    clip = transfx.slide_in(clip, duration=transition_duration, side='left')
            
            final_clips.append(clip)
        
        # Concatenate all clips
        logger.info("Concatenating clips...")
        video = concatenate_videoclips(final_clips, method="compose")
        
        # Add property title at the beginning
        if description:
            # Extract first line as title or use first few words
            title_text = description.split('\n')[0]
            if len(title_text) > 50:
                title_text = title_text[:50] + "..."
            
            title_clip = self._create_text_overlay(title_text, 'title', 5)
            
            # Get address or key info for subtitle
            subtitle_text = ""
            for line in description.split('\n')[:3]:
                if any(word in line.lower() for word in ['address', 'located', 'location']):
                    subtitle_text = line
                    break
            
            if not subtitle_text and len(description.split('\n')) > 1:
                subtitle_text = description.split('\n')[1]
                if len(subtitle_text) > 60:
                    subtitle_text = subtitle_text[:60] + "..."
            
            if subtitle_text:
                subtitle_clip = self._create_text_overlay(subtitle_text, 'subtitle', 5)
                video = CompositeVideoClip([video, title_clip, subtitle_clip])
            else:
                video = CompositeVideoClip([video, title_clip])
        
        # Add branding (logo)
        video = self._add_branding(video)
        
        # Add background music
        video = self._add_music(video, music)
        
        # Write the final video to file
        logger.info(f"Writing video to {output_path}...")
        video.write_videofile(output_path, fps=self.fps, codec='libx264', threads=4, audio_codec='aac')
        
        # Log completion time
        elapsed_time = time.time() - start_time
        logger.info(f"Video generation completed in {elapsed_time:.2f} seconds")
        
        return output_path


def main():
    """Entry point for the script."""
    parser = argparse.ArgumentParser(description="Generate real estate videos from images and description.")
    parser.add_argument("--images_dir", required=True, help="Directory containing property images")
    parser.add_argument("--description", required=True, help="Path to text file with property description")
    parser.add_argument("--output", required=True, help="Path to save the output video")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file")
    parser.add_argument("--duration", type=int, help="Video duration in seconds (overrides config)")
    parser.add_argument("--music", help="Music option to use (overrides random selection)")
    
    args = parser.parse_args()
    
    # Generate the video
    generator = RealEstateVideoGenerator(args.config)
    output_path = generator.generate_video(
        images_dir=args.images_dir,
        description_file=args.description,
        output_path=args.output,
        duration=args.duration,
        music=args.music
    )
    
    print(f"Video generated successfully: {output_path}")


if __name__ == "__main__":
    main()
