from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from datetime import datetime

def extract_location(filename: str) -> Tuple[float, float, float, float]:
    """Extract bbox coordinates from filename"""
    parts = filename.split('_')
    return (float(parts[1]), float(parts[2]), 
            float(parts[3]), float(parts[4]))

class DataOrganizer:
    def __init__(self, image_label_pairs: Dict[str, str]):
        self.pairs = image_label_pairs
        self.location_groups = self._group_by_location()
        
    def _group_by_location(self) -> Dict[Tuple[float, float, float, float], 
                                       List[Tuple[str, str, datetime]]]:
        """Group data pairs by location bbox"""
        groups = defaultdict(list)
        
        for img_path, label_path in self.pairs.items():
            img_name = Path(img_path).stem
            location = extract_location(img_name)
            
            # Extract datetime from image name
            date_str = img_name.split('_')[-2]
            time_str = img_name.split('_')[-1]
            dt = datetime.strptime(f"{date_str}_{time_str}", "%Y-%m-%d_%H%M")
            
            groups[location].append((img_path, label_path, dt))
        
        # Sort each group by datetime
        for location in groups:
            groups[location].sort(key=lambda x: x[2])
            
        return dict(groups)
    
    def get_locations(self) -> List[Tuple[float, float, float, float]]:
        """Get list of unique locations"""
        return list(self.location_groups.keys())
    
    def get_location_sequences(self, 
                             location: Tuple[float, float, float, float],
                             sequence_length: int = 5) -> List[List[Tuple[str, str]]]:
        """
        Get all possible sequences for a specific location
        Returns list of sequences, each sequence is a list of (image_path, label_path) pairs
        """
        location_data = self.location_groups[location]
        sequences = []
        
        for i in range(len(location_data) - sequence_length + 1):
            sequence = location_data[i:i + sequence_length]
            # Convert to list of (image_path, label_path) pairs
            sequence = [(s[0], s[1]) for s in sequence]
            sequences.append(sequence)
            
        return sequences
    
    def get_all_sequences(self, sequence_length: int = 5) -> Dict[Tuple[float, float, float, float], 
                                                                 List[List[Tuple[str, str]]]]:
        """Get sequences for all locations"""
        return {
            location: self.get_location_sequences(location, sequence_length)
            for location in self.get_locations()
        }
    
    def print_statistics(self):
        """Print summary statistics about the data organization"""
        print("\nData Organization Statistics:")
        print(f"Total locations: {len(self.location_groups)}")
        
        for location, sequences in self.location_groups.items():
            print(f"\nLocation {location}:")
            print(f"  Total timestamps: {len(sequences)}")
            date_range = sequences[-1][2] - sequences[0][2]
            print(f"  Date range: {date_range.days} days")
