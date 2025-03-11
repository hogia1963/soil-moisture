import rasterio
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm

def check_label_quality(label_path: str) -> Tuple[bool, float]:
    """
    Check if a label file has too many NaN values.
    Returns (is_valid, nan_percentage)
    """
    with rasterio.open(label_path) as src:
        data = src.read()
        total_pixels = np.prod(data.shape)
        nan_pixels = np.sum(np.isnan(data))
        nan_percentage = (nan_pixels / total_pixels) * 100
        return nan_percentage < 50, nan_percentage

def filter_data_pairs(image_label_pairs: Dict[str, str], 
                     max_nan_percentage: float = 50.0,
                     verbose: bool = True) -> Dict[str, str]:
    """
    Filter out image-label pairs where labels have too many NaN values.
    
    Args:
        image_label_pairs: Dictionary of image paths to label paths
        max_nan_percentage: Maximum allowed percentage of NaN values (0-100)
        verbose: Whether to print progress and statistics
    
    Returns:
        Filtered dictionary of image-label pairs
    """
    filtered_pairs = {}
    nan_stats = []
    
    if verbose:
        print(f"\nChecking label quality for {len(image_label_pairs)} pairs...")
        iterator = tqdm(image_label_pairs.items())
    else:
        iterator = image_label_pairs.items()
    
    for img_path, label_path in iterator:
        is_valid, nan_percentage = check_label_quality(label_path)
        nan_stats.append(nan_percentage)
        
        if is_valid:
            filtered_pairs[img_path] = label_path
    
    if verbose:
        total = len(image_label_pairs)
        kept = len(filtered_pairs)
        removed = total - kept
        
        print("\nData Cleaning Statistics:")
        print(f"Total pairs: {total}")
        print(f"Kept pairs: {kept} ({(kept/total)*100:.1f}%)")
        print(f"Removed pairs: {removed} ({(removed/total)*100:.1f}%)")
        print(f"Average NaN percentage: {np.mean(nan_stats):.1f}%")
        print(f"Median NaN percentage: {np.median(nan_stats):.1f}%")
    
    return filtered_pairs

def save_cleaning_report(image_label_pairs: Dict[str, str],
                        filtered_pairs: Dict[str, str],
                        output_dir: str = "data_cleaning_reports"):
    """Save a detailed report of the data cleaning process"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    report_path = output_dir / "cleaning_report.txt"
    removed_pairs = set(image_label_pairs.keys()) - set(filtered_pairs.keys())
    
    with open(report_path, "w") as f:
        f.write("Data Cleaning Report\n")
        f.write("===================\n\n")
        
        f.write("Summary:\n")
        f.write(f"Total pairs: {len(image_label_pairs)}\n")
        f.write(f"Kept pairs: {len(filtered_pairs)}\n")
        f.write(f"Removed pairs: {len(removed_pairs)}\n\n")
        
        f.write("Removed Files:\n")
        for img_path in sorted(removed_pairs):
            label_path = image_label_pairs[img_path]
            is_valid, nan_percentage = check_label_quality(label_path)
            f.write(f"- {Path(img_path).name}: {nan_percentage:.1f}% NaN values\n")

def get_clean_data_pairs(image_folder: str, 
                        label_folder: str,
                        max_nan_percentage: float = 50.0,
                        save_report: bool = True) -> Dict[str, str]:
    """
    Main function to get clean image-label pairs.
    
    Args:
        image_folder: Path to folder containing images
        label_folder: Path to folder containing labels
        max_nan_percentage: Maximum allowed percentage of NaN values
        save_report: Whether to save a cleaning report
    
    Returns:
        Dictionary of clean image-label pairs
    """
    from match_images_labels import match_images_with_labels
    
    # First match images with labels
    image_label_pairs = match_images_with_labels(image_folder, label_folder)
    
    if not image_label_pairs:
        raise ValueError("No image-label pairs found!")
    
    # Filter out pairs with bad labels
    filtered_pairs = filter_data_pairs(
        image_label_pairs, 
        max_nan_percentage=max_nan_percentage
    )
    
    if save_report:
        save_cleaning_report(image_label_pairs, filtered_pairs)
    
    return filtered_pairs
