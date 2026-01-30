import argparse
import torch
import gc
import logging
import sys
import os
import warnings
import json
from datetime import datetime

# Suppress warnings
warnings.filterwarnings("ignore")

# Import HeartLib
from heartlib import HeartTranscriptorPipeline

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
        logging.INFO: cyan + "%(asctime)s" + reset + " [" + green + "%(levelname)s" + reset + "] %(message)s",
        logging.ERROR: red + format_str + reset,
    }
    def format(self, record):
        return logging.Formatter(self.FORMATS.get(record.levelno), datefmt="%H:%M:%S").format(record)

logger = logging.getLogger("Transcriptor")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(ColoredFormatter())
logger.addHandler(ch)

def print_banner():
    print(f"\033[1;34m")
    print(r"""
  _    _                 _   _____                
 | |  | |               | | |_   _|               
 | |__| | ___  __ _ _ __| |_  | |_ __ __ _ _ __  
 |  __  |/ _ \/ _` | '__| __| | | '__/ _` | '_ \ 
 | |  | |  __/ (_| | |  | |_  | | | | (_| | | | |
 |_|  |_|\___|\__,_|_|   \__| |_|_|  \__,_|_| |_|
                                 INTEL XPU EDITION
    """)
    print(f"\033[0m")

def clean_memory():
    gc.collect()
    if hasattr(torch, 'xpu'): torch.xpu.empty_cache()

def main():
    print_banner()
    clean_memory()

    parser = argparse.ArgumentParser()
    parser.add_argument("--music_path", type=str, required=True, help="Path to mp3/wav file")
    parser.add_argument("--model_path", type=str, default="./ckpt", help="Path to checkpoint folder")
    parser.add_argument("--save_path", type=str, default=None, help="Output txt file path")
    args = parser.parse_args()

    # Hardware Check
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        device_name = torch.xpu.get_device_name(0)
        logger.info(f"Hardware: \033[1;33m{device_name}\033[0m")
        device = torch.device("xpu")
    else:
        logger.error("No Intel XPU detected! Falling back to CPU.")
        device = torch.device("cpu")

    # Validate Paths
    if not os.path.exists(args.music_path):
        logger.error(f"File not found: {args.music_path}")
        return
    
    # Load Pipeline
    logger.info("Loading HeartTranscriptor...")
    try:
        # Note: We pass the base ckpt folder; the pipeline class looks for 'HeartTranscriptor-oss' inside it
        pipe = HeartTranscriptorPipeline.from_pretrained(
            args.model_path,
            device=device,
            dtype=torch.float16 # Whisper usually handles FP16 well
        )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    logger.info(f"Transcribing: \033[1;36m{os.path.basename(args.music_path)}\033[0m")
    
    # Run Inference
    with torch.no_grad():
        # Using parameters optimized for singing voice
        result = pipe(
            args.music_path,
            max_new_tokens=256,
            num_beams=2,
            task="transcribe",
            condition_on_prev_tokens=False,
            compression_ratio_threshold=1.8,
            temperature=(0.0, 0.1, 0.2, 0.4),
            logprob_threshold=-1.0,
            no_speech_threshold=0.4,
            return_timestamps=True # Get timestamps too
        )

    # Output Handling
    text = result.get("text", "").strip()
    chunks = result.get("chunks", [])

    print("\n" + "="*30)
    print("TRANSCRIPTION RESULT")
    print("="*30)
    print(text)
    print("="*30 + "\n")

    # Save to file
    if args.save_path:
        out_file = args.save_path
    else:
        out_file = os.path.splitext(args.music_path)[0] + "_lyrics.txt"
    
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(f"Source: {os.path.basename(args.music_path)}\n")
        f.write(f"Date: {datetime.now().isoformat()}\n")
        f.write("-" * 20 + "\n\n")
        f.write(text + "\n\n")
        
        f.write("-" * 20 + "\n")
        f.write("TIMESTAMPS\n")
        f.write("-" * 20 + "\n")
        for chunk in chunks:
            ts = chunk.get('timestamp', (0,0))
            txt = chunk.get('text', '').strip()
            f.write(f"[{ts[0]:.2f}s -> {ts[1]:.2f}s]: {txt}\n")

    logger.info(f"Saved lyrics to: \033[1;35m{out_file}\033[0m")
    
    del pipe
    clean_memory()

if __name__ == "__main__":
    main()