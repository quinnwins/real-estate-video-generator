"""
Utility functions for processing real estate descriptions and extracting key information.
"""

import re
import spacy
import nltk
from nltk.tokenize import sent_tokenize
from typing import List, Dict, Tuple, Set

# Download required NLTK resources if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Error: Spacy model not found. Please install it with: python -m spacy download en_core_web_sm")


def extract_property_details(text: str) -> Dict[str, str]:
    """
    Extract key property details from description text.
    
    Args:
        text: Property description text
        
    Returns:
        Dictionary with property details
    """
    details = {}
    
    # Extract property type
    property_types = [
        "single family", "house", "apartment", "condo", "condominium", 
        "townhouse", "town home", "villa", "duplex", "triplex", "loft",
        "penthouse", "studio", "cottage", "bungalow", "mansion"
    ]
    
    for prop_type in property_types:
        if prop_type in text.lower():
            details["property_type"] = prop_type
            break
    
    # Extract number of bedrooms
    bedroom_pattern = r'(\d+)\s*(?:bed|bedroom|br\b)'
    bedroom_match = re.search(bedroom_pattern, text.lower())
    if bedroom_match:
        details["bedrooms"] = bedroom_match.group(1)
    
    # Extract number of bathrooms
    bathroom_pattern = r'(\d+(?:\.\d+)?)\s*(?:bath|bathroom|ba\b)'
    bathroom_match = re.search(bathroom_pattern, text.lower())
    if bathroom_match:
        details["bathrooms"] = bathroom_match.group(1)
    
    # Extract square footage
    sqft_pattern = r'(\d[\d,]+)\s*(?:sq\.?\s*ft\.?|square\s*feet|sqft)'
    sqft_match = re.search(sqft_pattern, text.lower())
    if sqft_match:
        # Remove commas from number
        sqft = sqft_match.group(1).replace(',', '')
        details["square_feet"] = sqft
    
    # Extract price if available
    price_pattern = r'\$\s*([\d,]+)'
    price_match = re.search(price_pattern, text)
    if price_match:
        price = price_match.group(1).replace(',', '')
        details["price"] = price
    
    # Extract address
    # This is a simplistic approach; addresses are complex and varied
    address_indicators = ["located at", "address", "located on"]
    lines = text.split('\n')
    
    for i, line in enumerate(lines):
        line_lower = line.lower()
        for indicator in address_indicators:
            if indicator in line_lower:
                # Extract the part after the indicator
                start_idx = line_lower.find(indicator) + len(indicator)
                possible_address = line[start_idx:].strip()
                if possible_address:
                    details["address"] = possible_address
                break
        
        # First line is often the address in listings
        if i == 0 and "address" not in details and not any(x in line_lower for x in ["welcome", "introducing", "price"]):
            details["address"] = line.strip()
    
    return details


def extract_key_features(text: str, max_features: int = 5) -> List[str]:
    """
    Extract key property features from the description.
    
    Args:
        text: Property description text
        max_features: Maximum number of features to extract
        
    Returns:
        List of key features as sentences
    """
    # Parse text with spaCy
    doc = nlp(text)
    
    # Break into sentences
    sentences = [sent.text.strip() for sent in doc.sents]
    
    # Common feature keywords in real estate
    feature_keywords = {
        # Interior features
        "kitchen", "granite", "stainless", "appliance", "counter", "cabinet",
        "bathroom", "master", "suite", "tub", "shower", "vanity",
        "hardwood", "floor", "carpet", "tile", "ceiling", "lighting", 
        "window", "natural light", "closet", "storage", "pantry",
        
        # Exterior features
        "yard", "garden", "landscape", "pool", "spa", "hot tub", "deck", 
        "patio", "balcony", "porch", "garage", "driveway", "parking",
        "fence", "gated", "security", "view", "panoramic",
        
        # Amenities
        "gym", "fitness", "clubhouse", "community", "park", "playground",
        "tennis", "basketball", "golf", "walking trail", "biking trail",
        
        # Qualities
        "new", "renovated", "updated", "modern", "contemporary", "luxury",
        "spacious", "open concept", "open floor plan", "custom", "high-end"
    }
    
    # Identify sentences with feature keywords
    feature_sentences = []
    for sentence in sentences:
        sentence_lower = sentence.lower()
        for keyword in feature_keywords:
            if keyword in sentence_lower and sentence not in feature_sentences:
                feature_sentences.append(sentence)
                break
    
    # Limit to max_features
    if len(feature_sentences) > max_features:
        # Score sentences by number of keywords
        scored_sentences = []
        for sentence in feature_sentences:
            sentence_lower = sentence.lower()
            count = sum(1 for keyword in feature_keywords if keyword in sentence_lower)
            scored_sentences.append((count, sentence))
        
        # Sort by score (descending) and take top max_features
        scored_sentences.sort(reverse=True)
        feature_sentences = [sentence for _, sentence in scored_sentences[:max_features]]
    
    return feature_sentences


def format_property_title(details: Dict[str, str]) -> str:
    """
    Create a formatted property title from extracted details.
    
    Args:
        details: Dictionary of property details
        
    Returns:
        Formatted property title
    """
    components = []
    
    # Add bedrooms/bathrooms if available
    if "bedrooms" in details and "bathrooms" in details:
        components.append(f"{details['bedrooms']} Bed, {details['bathrooms']} Bath")
    
    # Add property type if available
    if "property_type" in details:
        components.append(details["property_type"].title())
    
    # Add square footage if available
    if "square_feet" in details:
        components.append(f"{details['square_feet']} sq.ft.")
    
    # Add price if available
    if "price" in details:
        price_num = int(details["price"])
        if price_num >= 1000000:
            price_str = f"${price_num / 1000000:.1f}M"
        else:
            price_str = f"${price_num / 1000:.0f}K"
        components.append(price_str)
    
    # Add address if available (but keep it short)
    if "address" in details:
        address = details["address"]
        if len(address) > 30:
            address = address[:30] + "..."
        components.append(f"at {address}")
    
    # Join components
    return " | ".join(components)


def generate_text_overlays(description: str, max_overlays: int = 8) -> List[Dict[str, str]]:
    """
    Generate text overlays for the video from property description.
    
    Args:
        description: Property description text
        max_overlays: Maximum number of text overlays to generate
        
    Returns:
        List of dictionaries with text overlay content and type
    """
    overlays = []
    
    # Extract property details
    details = extract_property_details(description)
    
    # Generate title overlay
    title = format_property_title(details)
    overlays.append({
        "text": title,
        "type": "title"
    })
    
    # Add address as subtitle if available
    if "address" in details:
        overlays.append({
            "text": details["address"],
            "type": "subtitle"
        })
    
    # Extract key features
    features = extract_key_features(description, max_overlays - len(overlays))
    
    # Add features as body overlays
    for feature in features:
        overlays.append({
            "text": feature,
            "type": "body"
        })
    
    return overlays


if __name__ == "__main__":
    # Example usage
    sample_description = """
    123 Main Street, Anytown, USA
    
    Beautiful 4 bedroom, 2.5 bathroom single family home with 2,500 sq. ft. of living space. 
    
    This stunning property features a newly renovated kitchen with granite countertops and stainless steel appliances. 
    The spacious master suite includes a walk-in closet and a luxury bathroom with double vanity and soaking tub.
    
    Enjoy the beautiful backyard with a covered patio and professionally landscaped garden.
    
    Located in a highly desirable neighborhood with top-rated schools nearby.
    
    Offered at $850,000.
    """
    
    details = extract_property_details(sample_description)
    print("Extracted Property Details:")
    for key, value in details.items():
        print(f"  {key}: {value}")
    
    print("\nKey Features:")
    features = extract_key_features(sample_description)
    for feature in features:
        print(f"  • {feature}")
    
    print("\nFormatted Title:")
    title = format_property_title(details)
    print(f"  {title}")
    
    print("\nText Overlays:")
    overlays = generate_text_overlays(sample_description)
    for overlay in overlays:
        print(f"  Type: {overlay['type']}")
        print(f"  Text: {overlay['text']}")
        print()
