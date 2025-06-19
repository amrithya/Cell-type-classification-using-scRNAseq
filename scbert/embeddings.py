import os
import torch

ckpt_path = "/data1/data/corpus/scMODEL/finetune_model_Zheng68K.pkl"

if os.path.exists(ckpt_path):
    print(f"[INFO] Found checkpoint at {ckpt_path}. Loading...")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    
    # Check for expected keys
    expected_keys = ['epoch', 'model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict', 'losses']
    for key in expected_keys:
        if key in checkpoint:
            if checkpoint[key]:  # Check if not empty or None
                print(f"[OK] {key} is present and contains values.")
            else:
                print(f"[WARN] {key} is present but empty or None.")
        else:
            print(f"[MISSING] {key} not found in checkpoint.")

    # Example: print epoch
    if 'epoch' in checkpoint:
        print(f"[INFO] Saved epoch: {checkpoint['epoch']}")
else:
    print(f"[INFO] No checkpoint found at {ckpt_path}.")
