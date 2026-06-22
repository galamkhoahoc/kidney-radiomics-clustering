import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
KITS23_REPO_DIR = RAW_DATA_DIR / "kits23"

def run_command(cmd, cwd=None):
    """Utility function to run shell commands safely."""
    print(f"\n[INFO] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"[ERROR] Command failed with exit code {result.returncode}: {' '.join(cmd)}")
        sys.exit(1)
    print("[SUCCESS] Command completed.\n")

def main():
    print("=== Starting KiTS23 Dataset Setup ===")
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Clone KiTS23 repository
    if not KITS23_REPO_DIR.exists():
        print("[INFO] Cloning KiTS23 repository...")
        run_command(["git", "clone", "https://github.com/neheller/kits23", str(KITS23_REPO_DIR)])
    else:
        print(f"[INFO] KiTS23 repo already exists at {KITS23_REPO_DIR}. Skipping clone.")

    # 2. Install kits23 package via pip
    print("[INFO] Installing kits23 package in editable mode...")
    # Using sys.executable ensures it uses the same python/conda environment you run this script with
    run_command([sys.executable, "-m", "pip", "install", "-e", "."], cwd=KITS23_REPO_DIR)

    # 3. Download the actual dataset using the newly installed CLI
    print("[INFO] Downloading KiTS23 dataset...")
    # On a normal Linux server / Tmux (unlike Colab), the PATH is handled correctly by the OS,
    # so we don't need a restart here.
    kits_cmd = "kits23_download_data"
    
    # Fallback to absolute path of the script if not found in PATH directly 
    # (sometimes conda needs a shell re-source, but sys.executable's parent dir contains the script)
    bin_dir = Path(sys.executable).parent
    if (bin_dir / "kits23_download_data").exists():
        kits_cmd = str(bin_dir / "kits23_download_data")

    run_command([kits_cmd])
    
    print("=== Setup complete! You can now run `python main.py` ===")

if __name__ == "__main__":
    main()
