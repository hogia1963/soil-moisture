from pathlib import Path
from typing import Dict, Tuple, List
from datetime import datetime

def sort_tif_files_by_time(folder_path: str) -> Dict[Tuple[float, float, float, float], List[str]]:
    """
    Process all TIF files in a folder and sort them by time for each unique bounding box.
    
    Args:
        folder_path: Path to the folder containing TIF files with names in the format:
                    'combined_lat1_lon1_lat2_lon2_YYYY-MM-DD_HHMM.tif'
    
    Returns:
        Dictionary with bounding box coordinates as keys and sorted file paths as values.
        Bounding box key format: (lat1, lon1, lat2, lon2)
    """
    # Convert string path to Path object
    folder = Path(folder_path)
    
    # Check if folder exists
    if not folder.exists():
        raise ValueError(f"Folder not found: {folder_path}")
    if not folder.is_dir():
        raise ValueError(f"Path is not a directory: {folder_path}")
    
    # Dictionary to store files grouped by bounding box
    bbox_groups: Dict[Tuple[float, float, float, float], List[Tuple[datetime, str]]] = {}
    
    # Get all .tif files in the folder
    tif_files = list(folder.glob("*.tif"))
    
    if not tif_files:
        print(f"No TIF files found in {folder_path}")
        return {}
    
    for file_path in tif_files:
        try:
            # Get just the filename without extension
            filename = file_path.stem
            
            # Split the filename into components
            parts = filename.split('_')
            if len(parts) != 7:
                print(f"Skipping invalid filename format: {filename}")
                continue
                
            # Extract coordinates and time information
            lat1 = float(parts[1])
            lon1 = float(parts[2])
            lat2 = float(parts[3])
            lon2 = float(parts[4])
            date_str = parts[5]
            time_str = parts[6]
            
            # Create datetime object
            datetime_obj = datetime.strptime(f"{date_str}_{time_str}", "%Y-%m-%d_%H%M")
            
            # Create bounding box tuple
            bbox = (lat1, lon1, lat2, lon2)
            
            # Add to dictionary
            if bbox not in bbox_groups:
                bbox_groups[bbox] = []
            bbox_groups[bbox].append((datetime_obj, str(file_path)))
            
        except (ValueError, IndexError) as e:
            print(f"Error processing file {filename}: {str(e)}")
            continue
    
    # Sort files by time for each bounding box and create final dictionary
    sorted_groups = {}
    for bbox, file_list in bbox_groups.items():
        # Sort by datetime and extract only the file paths
        sorted_files = [file_path for _, file_path in sorted(file_list, key=lambda x: x[0])]
        sorted_groups[bbox] = sorted_files
    
    return sorted_groups

# Example usage:
def example_usage():
    folder_path = "/mnt/e/soil-moisture/soil-moisture/data/combine_tiffs"
    
    try:
        sorted_files = sort_tif_files_by_time(folder_path)
        
        if sorted_files:
            print(f"\nFound {len(sorted_files)} unique bounding boxes:")
            for bbox, file_list in sorted_files.items():
                print(f"\nBounding box {bbox}:")
                print(f"Number of files: {len(file_list)}")
                print("Files in chronological order:")
                for file_path in file_list:
                    print(f"  {file_path}")
        
    except ValueError as e:
        print(f"Error: {e}")
example_usage()