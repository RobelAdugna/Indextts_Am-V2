#!/usr/bin/env python3
"""
Smart Amharic Training Starter

This script automatically:
- Resumes training if a checkpoint exists
- Starts fresh training if no checkpoint found
- Verifies all required files before starting
- Shows clear status messages

Usage:
    python smart_train_amharic.py

For custom paths, edit the configuration section below.
"""

import subprocess
import sys
from pathlib import Path


# ============================================================================
# CONFIGURATION - Edit these paths to match your setup
# ============================================================================

CONFIG = {
    "output_dir": "trained_ckpts_fixed_fresh",
    "train_manifest": "preprocessed_amharic/train_pairs.jsonl",
    "val_manifest": "preprocessed_amharic/val_pairs.jsonl",
    "tokenizer": "tokenizers/amharic_extended_bpe.model",
    "config": "checkpoints/config.yaml",
    "base_checkpoint": "checkpoints/gpt.pth",
    
    # Training hyperparameters (optimized for Amharic with extended vocab)
    "learning_rate": "5e-6",      # Lower for stable embedding training
    "text_loss_weight": "0.4",    # Higher to prioritize text learning
    "mel_loss_weight": "0.6",     # Rebalanced
    "warmup_steps": "2000",       # Longer warmup for embeddings
    "epochs": "10",
    "log_interval": "100",
    "val_interval": "500",
    "save_interval": "1000",
    "keep_checkpoints": "3",
}

# ============================================================================


def print_header(title: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def verify_file(path: Path, description: str) -> bool:
    """Verify a file exists and print status."""
    if path.exists():
        print(f"  ✅ {description}: {path}")
        return True
    else:
        print(f"  ❌ {description} NOT FOUND: {path}")
        return False


def main() -> int:
    print_header("Smart Amharic Training Starter")
    
    # Convert string paths to Path objects
    output_dir = Path(CONFIG["output_dir"])
    train_manifest = Path(CONFIG["train_manifest"])
    val_manifest = Path(CONFIG["val_manifest"])
    tokenizer = Path(CONFIG["tokenizer"])
    config_file = Path(CONFIG["config"])
    base_checkpoint = Path(CONFIG["base_checkpoint"])
    latest_checkpoint = output_dir / "latest.pth"
    
    # Step 1: Verify required files
    print("[Step 1/4] Verifying required files...\n")
    all_files_ok = True
    all_files_ok &= verify_file(train_manifest, "Train manifest")
    all_files_ok &= verify_file(val_manifest, "Validation manifest")
    all_files_ok &= verify_file(tokenizer, "Tokenizer")
    all_files_ok &= verify_file(config_file, "Config")
    all_files_ok &= verify_file(base_checkpoint, "Base checkpoint")
    
    if not all_files_ok:
        print("\n❌ Error: Some required files are missing!")
        print("\nPlease ensure you have:")
        print("  1. Completed preprocessing (Tab 5 in WebUI)")
        print("  2. Generated prompt-target pairs (Tab 5.5 in WebUI)")
        print("  3. Extended the tokenizer (Tab 4 in WebUI)")
        print("  4. Downloaded base checkpoints (run download_requirements.bat)")
        return 1
    
    print("\n✅ All required files found!")
    
    # Step 2: Check for existing checkpoint
    print("\n[Step 2/4] Checking for existing checkpoint...\n")
    
    resume_mode = False
    if latest_checkpoint.exists():
        print(f"  ✅ Found checkpoint: {latest_checkpoint}")
        print(f"  📍 Will RESUME training from this checkpoint")
        resume_mode = True
    else:
        print(f"  🆕 No checkpoint found at: {latest_checkpoint}")
        print(f"  🚀 Will start FRESH training")
        resume_mode = False
    
    # Step 3: Build command
    print("\n[Step 3/4] Preparing training command...\n")
    
    cmd = [
        "python", "trainers/train_gpt_v2.py",
        "--train-manifest", str(train_manifest),
        "--val-manifest", str(val_manifest),
        "--tokenizer", str(tokenizer),
        "--config", str(config_file),
        "--base-checkpoint", str(base_checkpoint),
        "--output-dir", str(output_dir),
        "--epochs", CONFIG["epochs"],
        "--learning-rate", CONFIG["learning_rate"],
        "--text-loss-weight", CONFIG["text_loss_weight"],
        "--mel-loss-weight", CONFIG["mel_loss_weight"],
        "--warmup-steps", CONFIG["warmup_steps"],
        "--log-interval", CONFIG["log_interval"],
        "--val-interval", CONFIG["val_interval"],
        "--save-interval", CONFIG["save_interval"],
        "--keep-checkpoints", CONFIG["keep_checkpoints"],
        "--amp",
    ]
    
    if resume_mode:
        cmd.extend(["--resume", "auto"])
    
    # Show configuration
    print("  Configuration:")
    print(f"    Mode: {'RESUME' if resume_mode else 'FRESH START'}")
    print(f"    Output directory: {output_dir}")
    print(f"    Learning rate: {CONFIG['learning_rate']}")
    print(f"    Text loss weight: {CONFIG['text_loss_weight']}")
    print(f"    Mel loss weight: {CONFIG['mel_loss_weight']}")
    print(f"    Warmup steps: {CONFIG['warmup_steps']}")
    print(f"    Epochs: {CONFIG['epochs']}")
    
    # Step 4: Start training
    print("\n[Step 4/4] Starting training...")
    print("\nCommand:")
    print(" ".join(cmd))
    
    print_header("Training Output")
    
    try:
        result = subprocess.run(cmd)
        
        if result.returncode == 0:
            print_header("Training Completed Successfully! ✅")
            print(f"\nCheckpoints saved in: {output_dir}")
            print(f"\nTo resume this training later, simply run:")
            print(f"  python {sys.argv[0]}")
            print(f"\nTo view training progress:")
            print(f"  tensorboard --logdir {output_dir}/logs")
        else:
            print_header("Training Failed ❌")
            print(f"\nExit code: {result.returncode}")
            print(f"\nCheck the error messages above for details.")
        
        return result.returncode
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user (Ctrl+C)")
        print(f"\nTo resume, run: python {sys.argv[0]}")
        return 130  # Standard exit code for SIGINT
    
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
