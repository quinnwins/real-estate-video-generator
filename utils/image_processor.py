"""
Utility functions for processing and enhancing real estate images.
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from typing import Tuple, List, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def resize_to_aspect_ratio(img: Image.Image, target_width: int, target_height: int) -> Image.Image:
    """
    Resize image to match target aspect ratio with cropping if necessary.
    
    Args:
        img: PIL Image object
        target_width: Target width
        target_height: Target height
        
    Returns:
        Resized PIL Image object
    """
    # Calculate aspect ratios
    target_ratio = target_width / target_height
    img_ratio = img.width / img.height
    
    # If aspect ratios don't match (with small tolerance)
    if abs(img_ratio - target_ratio) > 0.01:
        if img_ratio > target_ratio:
            # Image is wider than target, crop width
            new_width = int(img.height * target_ratio)
            left = (img.width - new_width) // 2
            img = img.crop((left, 0, left + new_width, img.height))
        else:
            # Image is taller than target, crop height
            new_height = int(img.width / target_ratio)
            top = (img.height - new_height) // 2
            img = img.crop((0, top, img.width, top + new_height))
    
    # Resize to target dimensions
    return img.resize((target_width, target_height), Image.LANCZOS)


def enhance_image(img: Image.Image, config: Dict[str, float]) -> Image.Image:
    """
    Apply enhancements to improve image quality.
    
    Args:
        img: PIL Image object
        config: Dictionary with enhancement parameters
        
    Returns:
        Enhanced PIL Image object
    """
    # Adjust brightness
    if 'brightness_adjustment' in config:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(config['brightness_adjustment'])
    
    # Adjust contrast
    if 'contrast_adjustment' in config:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(config['contrast_adjustment'])
    
    # Adjust saturation
    if 'saturation_adjustment' in config:
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(config['saturation_adjustment'])
    
    # Apply subtle sharpening
    if 'sharpen' in config and config['sharpen']:
        img = img.filter(ImageFilter.SHARPEN)
    
    return img


def detect_image_quality(img_path: str) -> Dict[str, float]:
    """
    Analyze image quality to determine necessary enhancements.
    
    Args:
        img_path: Path to the image file
        
    Returns:
        Dictionary with quality metrics
    """
    try:
        # Open image with OpenCV for analysis
        img = cv2.imread(img_path)
        if img is None:
            logger.error(f"Failed to load image: {img_path}")
            return {
                'brightness': 0.5,
                'contrast': 0.5,
                'blur': 0.0,
                'noise': 0.0
            }
            
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Calculate brightness (0-1 scale)
        brightness = np.mean(gray) / 255.0
        
        # Calculate contrast
        contrast = gray.std() / 255.0
        
        # Detect blur using Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur = 1.0 - min(1.0, np.var(laplacian) / 500.0)  # Normalize
        
        # Estimate noise level
        # Split the image into patches and compute variance within patches
        noise = 0.0
        patch_size = 16
        if img.shape[0] > patch_size * 2 and img.shape[1] > patch_size * 2:
            # Use 5x5 grid of patches
            rows, cols = img.shape[:2]
            patch_vars = []
            
            for i in range(5):
                for j in range(5):
                    r_start = i * rows // 5
                    r_end = (i + 1) * rows // 5
                    c_start = j * cols // 5
                    c_end = (j + 1) * cols // 5
                    
                    patch = gray[r_start:r_end, c_start:c_end]
                    if patch.size > 0:
                        patch_vars.append(np.var(patch))
            
            if patch_vars:
                noise = min(1.0, np.mean(patch_vars) / 500.0)
        
        return {
            'brightness': brightness,
            'contrast': contrast,
            'blur': blur,
            'noise': noise
        }
    
    except Exception as e:
        logger.error(f"Error analyzing image quality: {e}")
        return {
            'brightness': 0.5,
            'contrast': 0.5,
            'blur': 0.0,
            'noise': 0.0
        }


def suggest_enhancements(quality_metrics: Dict[str, float]) -> Dict[str, float]:
    """
    Suggest image enhancements based on quality metrics.
    
    Args:
        quality_metrics: Dictionary with quality metrics
        
    Returns:
        Dictionary with suggested enhancement parameters
    """
    suggestions = {}
    
    # Brightness adjustment
    brightness = quality_metrics['brightness']
    if brightness < 0.4:
        # Too dark, increase brightness
        suggestions['brightness_adjustment'] = 1.0 + (0.4 - brightness) * 1.5
    elif brightness > 0.7:
        # Too bright, decrease brightness
        suggestions['brightness_adjustment'] = 1.0 - (brightness - 0.7) * 0.8
    else:
        # Brightness is fine, slight enhancement
        suggestions['brightness_adjustment'] = 1.05
    
    # Contrast adjustment
    contrast = quality_metrics['contrast']
    if contrast < 0.15:
        # Low contrast, increase
        suggestions['contrast_adjustment'] = 1.0 + (0.15 - contrast) * 3.0
    else:
        # Normal or high contrast, slight enhancement
        suggestions['contrast_adjustment'] = 1.1
    
    # Saturation adjustment (slight boost for real estate photos)
    suggestions['saturation_adjustment'] = 1.1
    
    # Sharpening if blurry
    if quality_metrics['blur'] > 0.5:
        suggestions['sharpen'] = True
    else:
        suggestions['sharpen'] = False
    
    return suggestions


def process_image(img_path: str, target_width: int, target_height: int) -> Image.Image:
    """
    Process an image for video creation.
    
    Args:
        img_path: Path to the image file
        target_width: Target width
        target_height: Target height
        
    Returns:
        Processed PIL Image object
    """
    try:
        # Analyze image quality
        quality_metrics = detect_image_quality(img_path)
        
        # Get enhancement suggestions
        enhancements = suggest_enhancements(quality_metrics)
        
        # Open and process image with PIL
        img = Image.open(img_path)
        
        # Ensure correct aspect ratio
        img = resize_to_aspect_ratio(img, target_width, target_height)
        
        # Apply enhancements
        img = enhance_image(img, enhancements)
        
        return img
    
    except Exception as e:
        logger.error(f"Error processing image {img_path}: {e}")
        # Return a blank image if processing fails
        return Image.new('RGB', (target_width, target_height), color='black')


def sort_images_by_quality(image_paths: List[str]) -> List[str]:
    """
    Sort images by quality for optimal sequence.
    
    Args:
        image_paths: List of image file paths
        
    Returns:
        Sorted list of image file paths
    """
    # Quality score function
    def quality_score(path):
        metrics = detect_image_quality(path)
        # Higher is better
        score = 0.0
        
        # Good brightness (not too dark, not too bright)
        brightness_score = 1.0 - abs(metrics['brightness'] - 0.55) * 2
        
        # Good contrast (higher is better, up to a point)
        contrast_score = min(1.0, metrics['contrast'] * 4.0)
        
        # Sharpness (less blur is better)
        sharpness_score = 1.0 - metrics['blur']
        
        # Weight the components
        score = brightness_score * 0.3 + contrast_score * 0.4 + sharpness_score * 0.3
        
        return score
    
    # Calculate scores and sort
    try:
        scored_images = [(path, quality_score(path)) for path in image_paths]
        scored_images.sort(key=lambda x: x[1], reverse=True)
        return [path for path, _ in scored_images]
    except Exception as e:
        logger.error(f"Error sorting images by quality: {e}")
        # Return original order if sorting fails
        return image_paths


def optimize_image_sequence(image_paths: List[str]) -> List[str]:
    """
    Optimize the sequence of images for a better video flow.
    
    Args:
        image_paths: List of image file paths
        
    Returns:
        Reordered list of image file paths
    """
    # First sort by quality to get best images up front
    sorted_paths = sort_images_by_quality(image_paths)
    
    # Categorize images by content type (using file names as hints)
    # This is a simplified approach - real implementation would use ML image classification
    categories = {
        'exterior': [],
        'kitchen': [],
        'living': [],
        'bedroom': [],
        'bathroom': [],
        'other': []
    }
    
    for path in sorted_paths:
        filename = os.path.basename(path).lower()
        
        if any(term in filename for term in ['exterior', 'outside', 'front', 'rear', 'back']):
            categories['exterior'].append(path)
        elif any(term in filename for term in ['kitchen', 'dining']):
            categories['kitchen'].append(path)
        elif any(term in filename for term in ['living', 'family', 'great']):
            categories['living'].append(path)
        elif any(term in filename for term in ['bed', 'master']):
            categories['bedroom'].append(path)
        elif any(term in filename for term in ['bath', 'shower']):
            categories['bathroom'].append(path)
        else:
            categories['other'].append(path)
    
    # Create a logical flow:
    # 1. Exterior
    # 2. Living areas
    # 3. Kitchen/Dining
    # 4. Bedrooms
    # 5. Bathrooms
    # 6. Other
    optimized_sequence = []
    optimized_sequence.extend(categories['exterior'])
    optimized_sequence.extend(categories['living'])
    optimized_sequence.extend(categories['kitchen'])
    optimized_sequence.extend(categories['bedroom'])
    optimized_sequence.extend(categories['bathroom'])
    optimized_sequence.extend(categories['other'])
    
    # Make sure we don't lose any images in the reordering
    if len(optimized_sequence) != len(sorted_paths):
        logger.warning("Image count mismatch after reordering. Using quality-sorted sequence instead.")
        return sorted_paths
    
    return optimized_sequence


def analyze_image_features(img_path: str) -> Dict[str, Any]:
    """
    Analyze image to extract features for dynamic effects selection.
    
    Args:
        img_path: Path to the image file
        
    Returns:
        Dictionary with image features
    """
    try:
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            return {'error': f"Failed to load image: {img_path}"}
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Get image dimensions
        height, width = gray.shape
        
        # Detect edges
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges > 0) / (width * height)
        
        # Calculate average color
        avg_color = np.mean(img, axis=(0, 1)).tolist()
        
        # Check if image has a dominant direction (horizontal or vertical lines)
        # Use Hough Line Transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
        horizontal_lines = 0
        vertical_lines = 0
        
        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    if abs(x2 - x1) > abs(y2 - y1):
                        horizontal_lines += 1
                    else:
                        vertical_lines += 1
        
        dominant_direction = "horizontal" if horizontal_lines > vertical_lines else "vertical"
        
        # Calculate brightness distribution across the image
        # Split image into 3x3 grid and calculate brightness in each cell
        grid_size = 3
        brightness_map = []
        for i in range(grid_size):
            row = []
            for j in range(grid_size):
                y1 = i * height // grid_size
                y2 = (i + 1) * height // grid_size
                x1 = j * width // grid_size
                x2 = (j + 1) * width // grid_size
                cell = gray[y1:y2, x1:x2]
                cell_brightness = np.mean(cell) / 255.0
                row.append(cell_brightness)
            brightness_map.append(row)
        
        # Detect if image has a central focus
        # Simple heuristic: is the center brighter/more contrasty than edges?
        center_brightness = brightness_map[1][1]
        edge_brightness = np.mean([
            brightness_map[0][0], brightness_map[0][2],
            brightness_map[2][0], brightness_map[2][2]
        ])
        has_central_focus = center_brightness > edge_brightness
        
        return {
            'width': width,
            'height': height,
            'aspect_ratio': width / height,
            'edge_density': edge_density,
            'avg_color': avg_color,
            'dominant_direction': dominant_direction,
            'brightness_map': brightness_map,
            'has_central_focus': has_central_focus
        }
        
    except Exception as e:
        logger.error(f"Error analyzing image features: {e}")
        return {'error': str(e)}


def suggest_dynamic_effect(image_features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Suggest appropriate dynamic effect for an image based on its features.
    
    Args:
        image_features: Dictionary with image features
        
    Returns:
        Dictionary with effect parameters
    """
    # Default effects
    effect = {
        'zoom_type': 'in',  # 'in', 'out', 'none'
        'zoom_factor': 1.1,
        'pan_direction': 'none',  # 'left', 'right', 'up', 'down', 'none'
        'pan_amount': 0.1
    }
    
    # Check if we have an error
    if 'error' in image_features:
        return effect
    
    # Adjust effects based on image features
    
    # For images with central focus, zoom in
    if image_features.get('has_central_focus', False):
        effect['zoom_type'] = 'in'
        effect['zoom_factor'] = 1.15
        effect['pan_direction'] = 'none'
    
    # For images with strong horizontal or vertical lines
    elif image_features.get('dominant_direction') == 'horizontal':
        # Horizontal dominant - pan left or right
        effect['zoom_type'] = 'none'
        effect['pan_direction'] = 'left' if random.random() > 0.5 else 'right'
        effect['pan_amount'] = 0.15
    
    elif image_features.get('dominant_direction') == 'vertical':
        # Vertical dominant - pan up or down
        effect['zoom_type'] = 'none'
        effect['pan_direction'] = 'up' if random.random() > 0.5 else 'down'
        effect['pan_amount'] = 0.15
    
    # For images with high edge density (lots of details), do a slight zoom out
    elif image_features.get('edge_density', 0) > 0.1:
        effect['zoom_type'] = 'out'
        effect['zoom_factor'] = 1.1
        effect['pan_direction'] = 'none'
    
    # For more uniform images, combine zoom and pan
    else:
        effect['zoom_type'] = 'in' if random.random() > 0.5 else 'out'
        directions = ['left', 'right', 'up', 'down']
        effect['pan_direction'] = random.choice(directions)
        effect['pan_amount'] = 0.1
    
    return effect


import random

if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        if os.path.exists(img_path):
            # Process the image
            target_width, target_height = 1920, 1080
            processed_img = process_image(img_path, target_width, target_height)
            
            # Save the processed image
            output_path = os.path.splitext(img_path)[0] + "_processed.jpg"
            processed_img.save(output_path, quality=95)
            print(f"Processed image saved to {output_path}")
            
            # Analyze and suggest effects
            features = analyze_image_features(img_path)
            effects = suggest_dynamic_effect(features)
            
            print("Image Features:")
            for key, value in features.items():
                if key != 'brightness_map':
                    print(f"  {key}: {value}")
            
            print("\nSuggested Dynamic Effect:")
            for key, value in effects.items():
                print(f"  {key}: {value}")
        else:
            print(f"Image file not found: {img_path}")
    else:
        print("Usage: python image_processor.py <image_path>")
