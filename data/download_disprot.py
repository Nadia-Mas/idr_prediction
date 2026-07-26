%%writefile /content/idr_prediction/data/download_disprot.py
"""
Download and parse DisProt data for IDR prediction.
Fetches full entry details to get disorder annotations.
"""

import requests
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from tqdm import tqdm
import os
import time
import json

class DisProtDownloader:
    def __init__(self, base_url: str = "https://disprot.org/api/v1"):
        self.base_url = base_url
        self.session = requests.Session()
        self.cache_dir = "data/cache"
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_entries(self, limit: int = 500) -> List[Dict]:
        """Fetch list of entries (only metadata, no annotations yet)."""
        url = f"{self.base_url}/entries"
        params = {"limit": limit}
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            entries = response.json()
            print(f"Fetched {len(entries)} entry summaries from DisProt")
            return entries
        except Exception as e:
            print(f"Error fetching entries: {e}")
            return []

    def get_entry_details(self, entry_id: str) -> Dict:
        """Fetch full details (including annotations) for a single entry."""
        cache_path = f"{self.cache_dir}/{entry_id}.json"
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                return json.load(f)
        
        url = f"{self.base_url}/entries/{entry_id}"
        try:
            response = self.session.get(url)
            response.raise_for_status()
            data = response.json()
            with open(cache_path, 'w') as f:
                json.dump(data, f)
            time.sleep(0.2)
            return data
        except Exception as e:
            print(f"Error fetching entry {entry_id}: {e}")
            return {}

    def parse_disorder_annotations(self, entry: Dict) -> List[Tuple[int, int]]:
        """Extract IDR regions from the full entry details."""
        idr_regions = []
        annotations = entry.get('annotations', [])
        for ann in annotations:
            if ann.get('type') == 'disorder':
                start = ann.get('start')
                end = ann.get('end')
                if start is not None and end is not None:
                    idr_regions.append((start - 1, end))
        return idr_regions

    def create_dataset(self, summaries: List[Dict], max_entries: int = 200) -> pd.DataFrame:
        data = []
        print(f"Fetching details for up to {max_entries} entries...")
        for i, summary in enumerate(tqdm(summaries[:max_entries], desc="Fetching details")):
            entry_id = summary.get('id')
            if not entry_id:
                continue
            full_entry = self.get_entry_details(entry_id)
            if not full_entry:
                continue

            sequence = full_entry.get('sequence', '')
            if not sequence or len(sequence) < 20:
                continue

            idr_regions = self.parse_disorder_annotations(full_entry)
            labels = np.zeros(len(sequence), dtype=int)
            for start, end in idr_regions:
                start = max(0, start)
                end = min(len(sequence), end)
                labels[start:end] = 1

            data.append({
                'entry_id': entry_id,
                'uniprot_id': full_entry.get('uniprot_id', ''),
                'sequence': sequence,
                'labels': labels.tolist(),
                'idr_regions': idr_regions,
                'length': len(sequence)
            })

        df = pd.DataFrame(data)
        print(f"Created dataset with {len(df)} sequences")
        return df

def main():
    downloader = DisProtDownloader()
    summaries = downloader.get_entries(limit=500)
    if not summaries:
        print("No entries fetched. Check internet connection.")
        return
    df = downloader.create_dataset(summaries, max_entries=200)
    os.makedirs("data/raw", exist_ok=True)
    df.to_pickle("data/raw/disprot_dataset.pkl")
    df.to_csv("data/raw/disprot_dataset.csv", index=False)
    total_idr = sum(df['labels'].apply(lambda x: sum(x)))
    print(f"\nDataset Summary:")
    print(f"  Total sequences: {len(df)}")
    print(f"  Total residues: {sum(df['length'])}")
    print(f"  IDR residues: {total_idr}")
    print(f"  IDR proportion: {total_idr / sum(df['length']):.2%}")

if __name__ == "__main__":
    main()
