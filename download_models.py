import argparse
import os
import sys
import logging
import importlib.util
from huggingface_hub import snapshot_download, hf_hub_download

# ==========================================
# UI & LOGGING
# ==========================================
class ColoredFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    cyan = "\x1b[36;20m"
    green = "\x1b[32;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format_str = "%(asctime)s - %(levelname)s - %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format_str + reset,
        logging.INFO: cyan + "%(asctime)s" + reset + " [" + green + "%(levelname)s" + reset + "] %(message)s",
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%H:%M:%S")
        return formatter.format(record)

logger = logging.getLogger("Downloader")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(ColoredFormatter())
logger.addHandler(ch)

def print_banner():
    print(f"\033[1;32m")
    print(r"""
  _    _                 _   __  __       _      
 | |  | |               | | |  \/  |     | |     
 | |__| | ___  __ _ _ __| |_| \  / |_   _| | __ _ 
 |  __  |/ _ \/ _` | '__| __| |\/| | | | | |/ _` |
 | |  | |  __/ (_| | |  | |_| |  | | |_| | | (_| |
 |_|  |_|\___|\__,_|_|   \__|_|  |_|\__,_|_|\__,_|
            TURBO DOWNLOAD MANAGER
    """)
    print(f"\033[0m")

# ==========================================
# REPO CONFIGURATION
# ==========================================
REPO_CODEC = "HeartMuLa/HeartCodec-oss"
REPO_3B = "HeartMuLa/HeartMuLa-oss-3B"
REPO_CONFIGS = "HeartMuLa/HeartMuLaGen" 

# ==========================================
# SPEED OPTIMIZATIONS
# ==========================================
def enable_speed_hacks(use_mirror):
    # 1. Check for hf_transfer (Rust-based downloader)
    if importlib.util.find_spec("hf_transfer"):
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        logger.info("🚀 \033[1;35mhf_transfer\033[0m detected and enabled! (Max Speed)")
    else:
        logger.warning("💡 Tip: Run \033[1;33mpip install hf_transfer\033[0m for 2x-4x faster downloads.")

    # 2. Configure Mirror
    if use_mirror:
        # If use_mirror is just "True" (flag present), use default mirror
        # If it's a string, use that URL
        mirror_url = "https://hf-mirror.com" if use_mirror is True else use_mirror
        os.environ["HF_ENDPOINT"] = mirror_url
        logger.info(f"🌐 Using Mirror: \033[1;36m{mirror_url}\033[0m")

# ==========================================
# DOWNLOAD FUNCTIONS
# ==========================================
def download_codec(target_dir, workers):
    path = os.path.join(target_dir, "HeartCodec-oss")
    logger.info(f"Downloading HeartCodec to: \033[1;33m{path}\033[0m")
    snapshot_download(
        repo_id=REPO_CODEC,
        local_dir=path,
        local_dir_use_symlinks=False,
        ignore_patterns=["*.git*"],
        max_workers=workers
    )
    logger.info("HeartCodec download complete.")

def download_3b(target_dir, workers):
    path = os.path.join(target_dir, "HeartMuLa-oss-3B")
    logger.info(f"Downloading HeartMuLa-3B to: \033[1;33m{path}\033[0m")
    snapshot_download(
        repo_id=REPO_3B,
        local_dir=path,
        local_dir_use_symlinks=False,
        ignore_patterns=["*.git*"],
        max_workers=workers
    )
    logger.info("HeartMuLa-3B download complete.")

def download_configs(target_dir):
    logger.info(f"Downloading Tokenizer & Configs to: \033[1;33m{target_dir}\033[0m")
    files_to_download = ["tokenizer.json", "gen_config.json"]
    for filename in files_to_download:
        hf_hub_download(
            repo_id=REPO_CONFIGS,
            filename=filename,
            local_dir=target_dir,
            local_dir_use_symlinks=False
        )
    logger.info("Configs download complete.")

def main():
    print_banner()
    parser = argparse.ArgumentParser(description="HeartMuLa Turbo Downloader")
    
    # Path
    parser.add_argument("--dir", type=str, default="./ckpt", help="Target directory (Default: ./ckpt)")
    
    # Speed Options
    parser.add_argument("--mirror", nargs="?", const=True, default=False, 
                        help="Use hf-mirror.com or provide custom URL")
    parser.add_argument("--workers", type=int, default=8, 
                        help="Number of parallel connections (Default: 8)")

    # Selection Groups
    group = parser.add_argument_group("Selection Options")
    group.add_argument("--all", action="store_true", help="Download EVERYTHING (Default)")
    group.add_argument("--codec", action="store_true", help="Download only HeartCodec")
    group.add_argument("--model3b", action="store_true", help="Download only HeartMuLa-3B Model")
    group.add_argument("--configs", action="store_true", help="Download only tokenizer.json and gen_config.json")
    
    args = parser.parse_args()

    # Apply Optimizations
    enable_speed_hacks(args.mirror)

    # Defaults
    if not (args.codec or args.model3b or args.configs):
        args.all = True

    target_dir = os.path.abspath(args.dir)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
        logger.info(f"Created directory: {target_dir}")

    try:
        if args.all or args.configs:
            download_configs(target_dir)
        
        if args.all or args.codec:
            download_codec(target_dir, args.workers)
            
        if args.all or args.model3b:
            download_3b(target_dir, args.workers)

        logger.info("\033[1;32mAll requested downloads finished successfully!\033[0m")
        logger.info(f"Models are ready in: {target_dir}")

    except Exception as e:
        logger.error(f"Download failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()