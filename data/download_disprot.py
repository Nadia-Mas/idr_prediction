"""
Download and parse DisProt data for IDR prediction.
DisProt is the manually curated database of intrinsically disordered proteins[reference:0][reference:1].
"""

import requests
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from Bio import SeqIO
from Bio.Seq import Seq
from tqdm import tqdm
import json
import os
import time

class DisProtDownloader:
    """Download and parse DisProt entries for IDR annotation."""
    
    def __init__(self, base_url: str = "https://disprot.org/api/v1"):
        self.base_url = base_url
        self.session = requests.Session()
        
    def get_entries(self, limit: int = 1000) -> List[Dict]:
        """Fetch DisProt entries."""
        url = f"{self.base_url}/entries"
        params = {"limit": limit}
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            entries = response.json()
            print(f"Fetched {len(entries)} entries from DisProt")
            return entries
        except Exception as e:
            print(f"Error fetching entries: {e}")
            return []
    
    def get_entry_details(self, entry_id: str) -> Dict:
        """Fetch detailed information for a specific entry."""
        url = f"{self.base_url}/entries/{entry_id}"
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching entry {entry_id}: {e}")
            return {}
    
    def parse_disorder_annotations(self, entry: Dict) -> List[Tuple[int, int]]:
        """Extract IDR annotations from entry."""
        idr_regions = []
        
        # Parse annotations from the entry
        annotations = entry.get('annotations', [])
        for ann in annotations:
            if ann.get('type') == 'disorder':
                start = ann.get('start', 0)
                end = ann.get('end', 0)
                if start and end:
                    idr_regions.append((start - 1, end))  # Convert to 0-indexed
                    
        return idr_regions
    
    def create_dataset(self, entries: List[Dict]) -> pd.DataFrame:
        """Create a DataFrame with sequences and IDR labels."""
        data = []
        
        for entry in tqdm(entries, desc="Processing entries"):
            sequence = entry.get('sequence', '')
            if not sequence or len(sequence) < 20:
                continue
                
            # Get IDR annotations
            idr_regions = self.parse_disorder_annotations(entry)
            
            # Create per-residue labels
            labels = np.zeros(len(sequence), dtype=int)
            for start, end in idr_regions:
                labels[start:end] = 1
                
            data.append({
                'entry_id': entry.get('id', ''),
                'uniprot_id': entry.get('uniprot_id', ''),
                'sequence': sequence,
                'labels': labels.tolist(),
                'idr_regions': idr_regions,
                'length': len(sequence)
            })
            
        df = pd.DataFrame(data)
        print(f"Created dataset with {len(df)} sequences")
        return df
    
    def save_dataset(self, df: pd.DataFrame, output_dir: str = "data/raw"):
        """Save dataset to disk."""
        os.makedirs(output_dir, exist_ok=True)
        df.to_pickle(f"{output_dir}/disprot_dataset.pkl")
        df.to_csv(f"{output_dir}/disprot_dataset.csv", index=False)
        print(f"Dataset saved to {output_dir}/")

def main():
    """Download and save DisProt dataset."""
    downloader = DisProtDownloader()
    
    # Fetch entries
    entries = downloader.get_entries(limit=2000)
    
    # Create dataset
    if entries:
        df = downloader.create_dataset(entries)
        downloader.save_dataset(df)
        
        # Print summary
        total_idr = sum(df['labels'].apply(lambda x: sum(x)))
        print(f"\nDataset Summary:")
        print(f"  Total sequences: {len(df)}")
        print(f"  Total residues: {sum(df['length'])}")
        print(f"  IDR residues: {total_idr}")
        print(f"  IDR proportion: {total_idr / sum(df['length']):.2%}")
    else:
        print("No entries fetched. Please check your internet connection.")

if __name__ == "__main__":
    main()
