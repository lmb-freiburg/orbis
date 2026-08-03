import os
import torch

def split_cache_file(input_file, train_output_file, val_output_file, split_ratio=0.8):
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' does not exist!")
        return

    print(f"\n==================================================")
    print(f"Splitting '{input_file}' into 80/20 Train/Val...")
    print(f"==================================================")

    data = torch.load(input_file)
    total_samples = len(data['features'])
    train_size = int(split_ratio * total_samples)
    val_size = total_samples - train_size

    print(f"Total Samples: {total_samples}")
    print(f"Train Split Size (80%): {train_size}")
    print(f"Val Split Size (20%):   {val_size}")

    train_data = {}
    val_data = {}

    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            train_data[k] = v[:train_size]
            val_data[k] = v[train_size:]
        elif isinstance(v, list):
            train_data[k] = v[:train_size]
            val_data[k] = v[train_size:]
        else:
            train_data[k] = v
            val_data[k] = v

    if 'features' in train_data and train_data['features'].dtype == torch.float32:
        train_data['features'] = train_data['features'].half()
    if 'features' in val_data and val_data['features'].dtype == torch.float32:
        val_data['features'] = val_data['features'].half()

    torch.save(train_data, train_output_file)
    torch.save(val_data, val_output_file)

    print(f"\nSaved Train Split to '{train_output_file}':")
    print(f"  - features: {train_data['features'].shape}")
    print(f"  - labels:   {train_data['labels'].shape}")
    print(f"  - keys:     {list(train_data.keys())}")

    print(f"\nSaved Val Split to '{val_output_file}':")
    print(f"  - features: {val_data['features'].shape}")
    print(f"  - labels:   {val_data['labels'].shape}")
    print(f"  - keys:     {list(val_data.keys())}")

if __name__ == "__main__":
    cache_dir = "./cached_features"
    
    # 1. Split Block 18
    b18_input = os.path.join(cache_dir, "train_block18_all_correct_unpooled_partial_mc.pt")
    b18_train_out = os.path.join(cache_dir, "train_block18_all_correct_unpooled_mc.pt")
    b18_val_out = os.path.join(cache_dir, "val_block18_all_correct_unpooled_mc.pt")
    split_cache_file(b18_input, b18_train_out, b18_val_out, split_ratio=0.8)

    # # 2. Split Block 20 if present
    # b20_input = os.path.join(cache_dir, "train_block20_all_correct_unpooled_partial_mc.pt")
    # b20_train_out = os.path.join(cache_dir, "train_block20_all_correct_unpooled_mc.pt")
    # b20_val_out = os.path.join(cache_dir, "val_block20_all_correct_unpooled_mc.pt")
    # if os.path.exists(b20_input):
    #     split_cache_file(b20_input, b20_train_out, b20_val_out, split_ratio=0.8)
