# PERFORMANCE_OPTIMIZATION — HeartMuLa (consolidated)

Purpose: single, focused performance & optimization guide for users and developers.

## Quick summary
- Checkpoint load: ~2.3s → 0.8–1.0s (memory-mapped loading)
- Generation speed: ~2.7 fps → 4.5–6.5 fps (XPU + mixed precision)
- GPU utilization: ~45% → 75–90% (longer sequences / tuning)

## Key optimizations (what matters)
- Native PyTorch XPU (PyTorch 2.10+): use torch.xpu.* APIs for inference optimizations
- Memory-mapped loading: set `low_cpu_mem_usage=True` for from_pretrained()
- Mixed precision: use `torch.autocast(device_type="xpu")` (BF16 for HeartMuLa, FP16 for Whisper)
- Minimize GPU synchronizations: avoid per-step `torch.xpu.synchronize()` in hot loops
- Pre-allocate hot-path buffers outside loops
- Launcher env vars: set SYCL and thread envs in `run_optimized.sh`

## Recommended commands
- Validate XPU setup:
  python validate_xpu_setup.py

- Run optimized generation:
  ./run_optimized.sh --lyrics ./inputs/lyrics/default.txt
  or
  python xpu_music_gen.py --lyrics ./inputs/lyrics/default.txt

- Run optimized transcription:
  ./transcribe_optimized.sh --music_path ./audio/song.mp3
  or
  python xpu_transcribe.py --music_path ./audio/song.mp3

## Tuning tips
1. Increase `--max_audio_length_ms` to improve GPU utilization for generation.  
2. Use `--cfg_scale 1.0` when you prefer speed over CFG-guided quality.  
3. For transcription, raise `--batch_size` (if VRAM allows) and tune `--chunk_length_s`.  
4. Close other GPU-heavy apps and ensure latest Intel drivers / Level Zero runtime.

## Monitoring & benchmarking
- Real-time GPU: watch -n 0.5 'xpu-smi dump -m 0,1,5' or `intel_gpu_top`  
- Quick benchmark (30s gen): time python xpu_music_gen.py --max_audio_length_ms 30000  
- Transcription RTF target: RTF ≈ 0.15–0.30 (good)

## Troubleshooting performance regressions
- Confirm PyTorch XPU is available and models load with `low_cpu_mem_usage=True`.  
- Check CPU bottlenecks (`htop`) — CPU-bound preprocessing reduces XPU throughput.  
- Ensure launcher scripts are used (they set SYCL env vars).  

## Developer notes (where to look in code)
- Generation runtime: `xpu_music_gen.py`  
- Transcription pipeline: `src/heartlib/pipelines/lyrics_transcription.py` and `xpu_transcribe.py`  
- Launcher scripts: `run_optimized.sh`, `transcribe_optimized.sh`

---
For the full original (pre-split) document see `COMPLETE_OPTIMIZATION_SUMMARY.md`.