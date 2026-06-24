import os
import shutil
import glob
import subprocess
import yaml
import re
import argparse

def set_nested_value(dic, keys, value):
    """Utility to set a value in a deeply nested dictionary using a list of keys."""
    for key in keys[:-1]:
        dic = dic.setdefault(key, {})
    dic[keys[-1]] = value

def run_orbis_test_pipeline(
    exp_dir: str, 
    base_val_config_path: str, 
    output_gif_dir: str, 
    num_gen_frames: int, 
    num_steps: int,
    config_overrides: dict = None,
    sequence_id: str = "unknown"
):
    """
    Runs an evaluation roll-out with dynamically updated, nested YAML configuration 
    parameters and stores all resulting visual sequences/GIFs in a dedicated folder.
    """
    os.makedirs(output_gif_dir, exist_ok=True)
    
    # 1. Load the original YAML configuration
    with open(base_val_config_path, 'r') as f:
        config_data = yaml.safe_load(f) or {}
        
    # 2. Inject nested parameter overrides dynamically
    if config_overrides:
        for complex_key, value in config_overrides.items():
            keys = complex_key.split('.')
            set_nested_value(config_data, keys, value)
            
    # Save to a temporary validation config file inside the experiment directory
    temp_config_path = os.path.abspath(f"temp_run_val_config_{sequence_id}.yaml")
    with open(temp_config_path, 'w') as f:
        yaml.safe_dump(config_data, f, default_flow_style=False)
        
    # 3. Build and execute the Orbis evaluation command
    cmd = [
        "python", "evaluate/rollout.py",
        "--exp_dir", exp_dir,
        "--val_config", temp_config_path,
        "--num_gen_frames", str(num_gen_frames),
        "--num_steps", str(num_steps)
    ]
    
    print(f"\n[Seq {sequence_id}] Running rollout evaluation...")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during execution on sequence {sequence_id}: {e.stderr}")
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)
        return False

    # 4. Harvest generated media sequences (GIFs/Videos) 
    search_pattern = os.path.join(exp_dir, "**", "*.gif")
    generated_media = glob.glob(search_pattern, recursive=True)
    
    if not generated_media:
        search_pattern = os.path.join(exp_dir, "**", "*.mp4")
        generated_media = glob.glob(search_pattern, recursive=True)

    # 5. Copy and label with Sequence ID
    for file_path in generated_media:
        ext = os.path.splitext(file_path)[1]
        unique_name = f"sequence_{sequence_id}{ext}"
        dest_path = os.path.join(output_gif_dir, unique_name)
        
        shutil.copy2(file_path, dest_path)
        print(f" Saved pipeline artifact to: {dest_path}")

    # Cleanup temporary configuration file
    if os.path.exists(temp_config_path):
        os.remove(temp_config_path)
        
    return True


# --- Outer Processing Loop Wrapper ---
if __name__ == "__main__":
    # 1. Parse num_gen_frames and num_steps directly from the command line interface
    parser = argparse.ArgumentParser(description="Batch Orbis testing pipeline wrapper.")
    parser.add_argument("--num_gen_frames", type=int, required=True, help="Number of frames to generate during rollout.")
    parser.add_argument("--num_steps", type=int, required=True, help="Number of sampling evaluation steps.")
    args = parser.parse_args()

    EXPERIMENT_DIR = "logs_wm/orbis_288x512"
    BASE_VAL_YAML = EXPERIMENT_DIR+"/config.yaml"
    PIPELINE_GIF_STORE = "../Carla_wildlife_Results"
    DATASET_ROOT = "../carla_wildlife_sequences/raw_data"
    
    sequence_dirs = sorted(glob.glob(os.path.join(DATASET_ROOT, "sequence_*")))
    
    if not sequence_dirs:
        sequence_dirs = sorted([os.path.join(DATASET_ROOT, d) for d in os.listdir(DATASET_ROOT) 
                                if os.path.isdir(os.path.join(DATASET_ROOT, d))])

    print(f"Discovered {len(sequence_dirs)} sequences for evaluation.")

    for seq_path in sequence_dirs:
        folder_name = os.path.basename(seq_path)
        match = re.search(r'\d+', folder_name)
        seq_id = match.group(0) if match else folder_name
        
        image_extensions = ('*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG')
        all_frames = []
        for ext in image_extensions:
            all_frames.extend(glob.glob(os.path.join(seq_path, ext)))
            
        all_frames = sorted(all_frames)
        
        if len(all_frames) < 5:
            print(f"⚠️ Skipping Sequence {seq_id}: Found only {len(all_frames)} frames (minimum 5 required).")
            continue
            
        last_5_frames = all_frames[-5:]
        
        # 2. Print the titles/filenames of the 5 input context frames along with the current sequence ID
        print(f"\n" + "="*60)
        print(f"🚀 Processing Sequence ID: {seq_id}")
        print("Using context frames:")
        for frame in last_5_frames:
            print(f"  - {os.path.basename(frame)}")
        print("="*60)
        
        custom_pipeline_overrides = {
            "model.params.generator_config.params.max_num_frames": 5,
            "data.params.validation.params.image_paths": last_5_frames
        }
        
        run_orbis_test_pipeline(
            exp_dir=EXPERIMENT_DIR,
            base_val_config_path=BASE_VAL_YAML,
            output_gif_dir=PIPELINE_GIF_STORE,
            num_gen_frames=args.num_gen_frames, 
            num_steps=args.num_steps,
            config_overrides=custom_pipeline_overrides,
            sequence_id=seq_id
        )

    print("\n All sequence evaluations are complete.")