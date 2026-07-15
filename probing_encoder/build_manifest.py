import os
import json
from pathlib import Path

def generate_manifest(dota_dir="DoTA_prepared", annotations_dir="annotations", output_file="DoTA_prepared/manifest_1500_linear_probe.json"):
    dota_path = Path(dota_dir)
    anno_path = Path(annotations_dir)
    
    # Ensure the output directory exists
    dota_path.mkdir(parents=True, exist_ok=True)
    
    manifest_data = []
    
    print(f"Scanning {dota_path} for clips...")
    
    # Iterate through all directories in DoTA_prepared
    for clip_folder in dota_path.iterdir():
        if not clip_folder.is_dir():
            continue
            
        clip_id = clip_folder.name
        
        # Safety Check: Ensure this clip actually has both ood and non-ood folders prepared
        if not (clip_folder / "ood").exists() or not (clip_folder / "non-ood").exists():
            print(f"  [Skipping] {clip_id} - Missing 'ood' or 'non-ood' subfolders.")
            continue
            
        # Base dictionary that our PyTorch script requires
        clip_metadata = {
            "clip_id": clip_id,
            "accident_name": "unknown",
            "night": False,
            "ego_involve": False
        }
        
        # Enrich the manifest with data from the annotations folder (if it exists)
        anno_file = anno_path / f"{clip_id}.json"
        if anno_file.exists():
            with open(anno_file, "r") as f:
                try:
                    anno_data = json.load(f)
                    clip_metadata["accident_name"] = anno_data.get("accident_name", "unknown")
                    clip_metadata["night"] = anno_data.get("night", False)
                    clip_metadata["ego_involve"] = anno_data.get("ego_involve", False)
                except json.JSONDecodeError:
                    print(f"  [Warning] Could not read JSON for {clip_id}")
        else:
             print(f"  [Warning] No annotation file found for {clip_id}, using default metadata.")

        manifest_data.append(clip_metadata)
        
    # Write the combined data to the manifest file
    with open(output_file, "w") as f:
        json.dump(manifest_data, f, indent=2)
        
    print("\n--- Manifest Generation Complete ---")
    print(f"Total Valid Clips Found: {len(manifest_data)}")
    print(f"Manifest saved to: {output_file}")

if __name__ == "__main__":
    generate_manifest()