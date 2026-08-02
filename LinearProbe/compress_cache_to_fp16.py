import os
import torch

def compress_file_to_fp16(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    old_size = os.path.getsize(filepath) / (1024 * 1024) # MB
    data = torch.load(filepath, map_location='cpu')

    if 'features' in data and data['features'].dtype == torch.float32:
        data['features'] = data['features'].half()
        torch.save(data, filepath)
        new_size = os.path.getsize(filepath) / (1024 * 1024) # MB
        print(f"Compressed '{os.path.basename(filepath)}': {old_size:.1f} MB -> {new_size:.1f} MB (Saved {old_size - new_size:.1f} MB)")
    else:
        print(f"Skipped '{os.path.basename(filepath)}': already fp16 or no features key ({old_size:.1f} MB)")

if __name__ == "__main__":
    cache_dir = "./cached_features"
    files = sorted([os.path.join(cache_dir, f) for f in os.listdir(cache_dir) if f.endswith('.pt')])
    
    print(f"Compressing {len(files)} cache files in '{cache_dir}' to fp16...")
    for f in files:
        compress_file_to_fp16(f)
