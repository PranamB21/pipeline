# GPU-Accelerated Multimodal Data Organization

This project builds a GPU-accelerated multimodal organization and benchmarking pipeline for research workflows.

## Base Paths

- Pipeline: /media/kart/Laksh 320GB/PIPELINE/
- Input dataset: /media/kart/Laksh 320GB/UNORGANIZED/
- Organized output: /media/kart/Laksh 320GB/ORGANIZED/
- Results: /media/kart/Laksh 320GB/PIPELINE/results/

## Dataset Description

The input folder is a flat, unorganized directory containing 17,073 files mixed across modalities:

- MP4 videos
- WAV audio files
- TXT and JSON text files
- XLSX metadata spreadsheets

The pipeline scans recursively, detects supported types, benchmarks CPU and GPU processing time, and copies each file into an organized structure:

- /media/kart/Laksh 320GB/ORGANIZED/videos/
- /media/kart/Laksh 320GB/ORGANIZED/audio/
- /media/kart/Laksh 320GB/ORGANIZED/text/
- /media/kart/Laksh 320GB/ORGANIZED/metadata/

## Tools Used

- NVIDIA DALI (GPU video decode/processing)
- cuDF (GPU dataframe processing)
- CuPy (CUDA runtime verification)
- Torchaudio + PyTorch CUDA (GPU audio processing)
- pandas + openpyxl (XLSX ingest baseline)
- OpenCV (CPU video baseline)
- matplotlib (publication-ready benchmarking plots)

## How To Run

From Ubuntu terminal:

```bash
cd "/media/kart/Laksh 320GB/PIPELINE"
chmod +x setup.sh
./setup.sh
source "/media/kart/Laksh 320GB/PIPELINE/venv/bin/activate"
python main.py
python benchmark.py
```

## Expected Output

After execution:

1. Files are copied into modality folders under /media/kart/Laksh 320GB/ORGANIZED/
2. Benchmark summary CSV is generated:
   - /media/kart/Laksh 320GB/PIPELINE/results/benchmark.csv
3. Publication artifacts are generated:
   - /media/kart/Laksh 320GB/PIPELINE/results/speedup_table.csv
   - /media/kart/Laksh 320GB/PIPELINE/results/cpu_vs_gpu_time_per_filetype.png
   - /media/kart/Laksh 320GB/PIPELINE/results/speedup_vs_number_of_files.png

## Notes

- The scripts are configured for Python 3.11, CUDA 12.x, and NVIDIA GPU hosts.
- If GPU libraries fail to initialize, the benchmark logs corresponding errors per file type.
- The organization logic preserves all files and avoids collisions by appending suffixes when duplicate names exist.
