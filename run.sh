#!/bin/bash
# XPU Music Generator Shortcut

# Ensure we are in the script directory
cd "$(dirname "$0")"

# Run the python script
# You can change the --lyrics or --tags arguments here directly
python xpu_music_gen.py \
    --model_path "./ckpt" \
    --lyrics "./inputs/lyrics/echoes.txt" \
    --tags "./inputs/tags/sad_mood.txt" \
    --save_path "./output/echoes_of_silence.mp3" \
    --max_audio_length_ms 180000

# Pause to let user see the result
read -p "Press enter to close..."