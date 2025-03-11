from pathlib import Path
from typing import Dict, Tuple, List
from datetime import datetime, timedelta

def get_label_datetime(image_datetime: datetime) -> datetime:
    """
    Calculate the expected label datetime (image time + 1.5 hours)
    """
    return image_datetime + timedelta(hours=1, minutes=30)

def get_datetime_from_image(filename: str) -> Tuple[datetime, Tuple[float, float, float, float]]:
    """
    Extract datetime and bounding box from image filename
    Format: combined_lat1_lon1_lat2_lon2_YYYY-MM-DD_HHMM
    """
    parts = filename.split('_')
    if len(parts) != 7 or not parts[0] == 'combined':
        raise ValueError(f"Invalid image filename format: {filename}")
        
    # Extract coordinates and time information
    lat1 = float(parts[1])
    lon1 = float(parts[2])
    lat2 = float(parts[3])
    lon2 = float(parts[4])
    date_str = parts[5]
    time_str = parts[6]
    
    # Create datetime object and bbox tuple
    datetime_obj = datetime.strptime(f"{date_str}_{time_str}", "%Y-%m-%d_%H%M")
    bbox = (lat1, lon1, lat2, lon2)
    
    return datetime_obj, bbox

def get_datetime_from_label(filename: str) -> Tuple[datetime, Tuple[float, float, float, float]]:
    """
    Extract datetime and bounding box from label filename
    Format: smap_labels_lat1_lon1_lat2_lon2_YYYY-MM-DD_HHMM
    """
    parts = filename.split('_')
    if len(parts) != 8 or parts[0] != 'smap' or parts[1] != 'labels':
        raise ValueError(f"Invalid label filename format: {filename}")
        
    # Extract coordinates and time information
    lat1 = float(parts[2])
    lon1 = float(parts[3])
    lat2 = float(parts[4])
    lon2 = float(parts[5])
    date_str = parts[6]
    time_str = parts[7]
    
    # Create datetime object and bbox tuple
    datetime_obj = datetime.strptime(f"{date_str}_{time_str}", "%Y-%m-%d_%H%M")
    bbox = (lat1, lon1, lat2, lon2)
    
    return datetime_obj, bbox

def match_images_with_labels(image_folder: str, label_folder: str) -> Dict[str, str]:
    """
    Match each image with its corresponding label file from separate folders.
    Image format: combined_lat1_lon1_lat2_lon2_YYYY-MM-DD_HHMM.tif
    Label format: smap_labels_lat1_lon1_lat2_lon2_YYYY-MM-DD_HHMM.tif
    Label timestamp is 1.5 hours after the image time.
    
    Args:
        image_folder: Path to the folder containing image TIF files
        label_folder: Path to the folder containing label TIF files
    
    Returns:
        Dictionary with image file paths as keys and corresponding label file paths as values.
        If no label is found for an image, it won't be included in the dictionary.
    """
    # Convert string paths to Path objects
    image_path = Path(image_folder)
    label_path = Path(label_folder)
    
    # Check if folders exist
    for folder, name in [(image_path, "Image"), (label_path, "Label")]:
        if not folder.exists():
            raise ValueError(f"{name} folder not found: {folder}")
        if not folder.is_dir():
            raise ValueError(f"{name} path is not a directory: {folder}")
    
    # Get all .tif files from both folders
    image_files = list(image_path.glob("*.tif"))
    label_files = list(label_path.glob("*.tiff"))
    
    if not image_files:
        print(f"No TIF files found in image folder {image_folder}")
        return {}
    if not label_files:
        print(f"No TIF files found in label folder {label_folder}")
        return {}
    
    # Create dictionaries to store file information
    image_info = {}  # Store datetime and bbox for image files
    label_info = {}  # Store datetime and bbox for label files
    matches = {}     # Store matched image-label pairs
    
    # Process image files
    print(f"Processing {len(image_files)} image files...")
    for file_path in image_files:
        try:
            datetime_obj, bbox = get_datetime_from_image(file_path.stem)
            image_info[str(file_path)] = (datetime_obj, bbox)
        except ValueError as e:
            print(f"Error processing image file {file_path.name}: {str(e)}")
            continue
    
    # Process label files
    print(f"Processing {len(label_files)} label files...")
    for file_path in label_files:
        try:
            datetime_obj, bbox = get_datetime_from_label(file_path.stem)
            label_info[str(file_path)] = (datetime_obj, bbox)
        except ValueError as e:
            print(f"Error processing label file {file_path.name}: {str(e)}")
            continue
    
    # Match images with labels
    for image_path, (image_datetime, image_bbox) in image_info.items():
        expected_label_datetime = get_label_datetime(image_datetime)
        
        # Look for matching label
        for label_path, (label_datetime, label_bbox) in label_info.items():
            if (label_datetime == expected_label_datetime and 
                label_bbox == image_bbox):
                matches[image_path] = label_path
                break
    
    # Print summary
    print(f"\nFound {len(matches)} matches out of {len(image_files)} images")
    unmatched = len(image_files) - len(matches)
    if unmatched > 0:
        print(f"Warning: {unmatched} images have no matching labels")
    
    return matches

# Example usage:
def example_usage():
    image_folder = "/mnt/e/soil-moisture/soil-moisture/data/combine_tiffs"
    label_folder = "/mnt/e/soil-moisture/soil-moisture/data/label_tiffs"
    
    try:
        matches = match_images_with_labels(image_folder, label_folder)
        
        if matches:
            print("\nMatched pairs:")
            for image_path, label_path in matches.items():
                print(f"\nImage: {Path(image_path).name}")
                print(f"Label: {Path(label_path).name}")
        
    except ValueError as e:
        print(f"Error: {e}")
example_usage()