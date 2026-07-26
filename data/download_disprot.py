import requests
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from tqdm import tqdm
import os
import time

class DisProtDownloader:
    def __init__(self, base_url: str = "https://disprot.org/api"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def get_all_ids(self) -> List[str]:
        """Get list of all DisProt IDs using the /list_ids endpoint."""
        url = f"{self.base_url}/list_ids"
        try:
            response = self.session.get(url)
            response.raise_for_status()
            data = response.json()
            # The response is a dict with a 'disprot_ids' key
            ids = data.get('disprot_ids', [])
            print(f"Found {len(ids)} DisProt entries")
            return ids
        except Exception as e:
            print(f"Error fetching ID list: {e}")
            return []
    
    def get_entry(self, entry_id: str) -> Dict:
        """Fetch a single entry by its DisProt ID using the /{identifier} endpoint."""
        url = f"{self.base_url}/{entry_id}"
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching entry {entry_id}: {e}")
            return {}
    
    def parse_disorder_annotations(self, entry: Dict) -> List[Tuple[int, int]]:
        """Extract IDR regions from a DisProt entry."""
        idr_regions = []
        
        # In DisProt 9, regions are under 'regions' key
        regions = entry.get('regions', [])
        for region in regions:
            # Look for disorder annotations
            if region.get('is_disordered', False):
                start = region.get('start', 0)
                end = region.get('end', 0)
                if start and end:
                    idr_regions.append((start - 1, end))  # Convert to 0-indexed
        
        return idr_regions
    
    def create_dataset(self, ids: List[str], max_entries: int = 500) -> pd.DataFrame:
        """Create a dataset by fetching entries one by one."""
        data = []
        
        # Limit to a reasonable number for a quick run
        ids_to_fetch = ids[:max_entries]
        
        for entry_id in tqdm(ids_to_fetch, desc="Fetching entries"):
            entry = self.get_entry(entry_id)
            if not entry:
                continue
            
            # Get sequence from the entry
            sequence = entry.get('sequence', '')
            if not sequence or len(sequence) < 20:
                continue
            
            # Get disorder annotations
            idr_regions = self.parse_disorder_annotations(entry)
            
            # Create per-residue labels
            labels = np.zeros(len(sequence), dtype=int)
            for start, end in idr_regions:
                # Clamp to sequence length
                start = max(0, start)
                end = min(len(sequence), end)
                labels[start:end] = 1
            
            data.append({
                'entry_id': entry_id,
                'uniprot_id': entry.get('uniprot_accession', ''),
                'sequence': sequence,
                'labels': labels.tolist(),
                'idr_regions': idr_regions,
                'length': len(sequence)
            })
            
            # Be gentle to the server
            time.sleep(0.1)
        
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
    downloader = DisProtDownloader()
    
    # Step 1: Get list of all DisProt IDs
    ids = downloader.get_all_ids()
    
    if not ids:
        print("No IDs fetched. Please check your internet connection.")
        return
    
    # Step 2: Fetch entries (limit to 500 for a reasonable dataset)
    df = downloader.create_dataset(ids, max_entries=500)
    
    # Step 3: Save
    downloader.save_dataset(df)
    
    # Summary
    if len(df) > 0:
        total_idr = sum(df['labels'].apply(lambda x: sum(x)))
        total_residues = sum(df['length'])
        print(f"\nDataset Summary:")
        print(f"  Total sequences: {len(df)}")
        print(f"  Total residues: {total_residues}")
        print(f"  IDR residues: {total_idr}")
        print(f"  IDR proportion: {total_idr / total_residues:.2%}")
    else:
        print("No data was collected.")

if __name__ == "__main__":
    main()
