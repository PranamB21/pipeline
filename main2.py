#!/usr/bin/env python3
"""
GPU-accelerated multimodal data organizer.

This script scans a large, mixed dataset in a flat directory and:
1) Detects supported file types (.mp4, .wav, .txt, .json, .xlsx)
2) Runs both CPU and GPU processing paths per file type for benchmark timing
3) Copies each file into an organized output tree by modality
4) Writes per-type benchmark summary to benchmark.csv

All project paths are absolute and rooted at /media/kart/Laksh 320GB.

Batch processing notes
----------------------
Files are processed in batches of BATCH_SIZE (default 50) to allow GPU
libraries to amortize initialization and memory-transfer overhead across
multiple files per call:

- Video (DALI):  pipeline built once per batch, not once per file.
                 A binary-search fallback isolates bad files without
                 slowing down good files in the same batch.
- Audio (torch): waveforms transferred to GPU in a loop; single
                 cuda.synchronize() at end of batch.
- Text (cuDF):   all files in the batch are concatenated into one
                 DataFrame before the GPU aggregation.
- XLSX (cuDF):   all spreadsheets batch-read with pandas, then a single
                 cudf.from_pandas() covers the whole batch.

GPU warm-up
-----------
_warmup_gpu() is called once before the main processing loop.  Without
this, the CUDA context + cuDF runtime initialization (~3-10 s) would be
charged to the first file timed, inflating GPU measurements for that
extension type.
"""

from __future__ import annotations

import json
import shutil
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import pandas as pd

# -------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
BASE_DIR = Path("/run/media/pranam/Laksh 320GB")
INPUT_DIR  = BASE_DIR / "gpu_pipeline" / "UNORGANIZED"
_LOCAL = Path("/home/pranam/Downloads/PIPELINE")
OUTPUT_DIR   = BASE_DIR / "gpu_pipeline" / "ORGANIZED"
RESULTS_DIR  = _LOCAL / "results"
BENCHMARK_CSV = RESULTS_DIR / "benchmark.csv"

# How many files to process in each CPU/GPU batch call.
BATCH_SIZE = 50

# Map extension to target organized subfolder.
EXTENSION_TO_FOLDER = {
    "mp4": "videos",
    "wav": "audio",
    "txt": "text",
    "json": "text",
    "xlsx": "metadata",
}
SUPPORTED_EXTENSIONS = set(EXTENSION_TO_FOLDER.keys())

# ---------------------------------------------------------------------------
# Optional GPU imports — fail gracefully so CPU-only runs still work.
# ---------------------------------------------------------------------------
try:
    import cv2
except Exception:
    cv2 = None

try:
    import torch
    import torchaudio
except Exception:
    torch = None
    torchaudio = None

try:
    import cudf
except Exception:
    cudf = None

try:
    from nvidia.dali import fn
    from nvidia.dali import types
    from nvidia.dali.pipeline import Pipeline
except Exception:
    fn = None
    types = None
    Pipeline = None


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def ensure_output_dirs() -> None:
    """Create expected output/result directories once before processing."""
    for sub in ["videos", "audio", "text", "metadata"]:
        (OUTPUT_DIR / sub).mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def detect_extension(path: Path) -> str | None:
    """Return normalized extension without dot if supported, else None."""
    ext = path.suffix.lower().lstrip(".")
    return ext if ext in SUPPORTED_EXTENSIONS else None


def discover_files(root: Path) -> dict[str, list[Path]]:
    """Recursively discover supported files and group by extension."""
    grouped: dict[str, list[Path]] = defaultdict(list)
    for item in root.rglob("*"):
        if not item.is_file():
            continue
        ext = detect_extension(item)
        if ext:
            grouped[ext].append(item)
    return grouped


def unique_destination_path(destination_dir: Path, source_file: Path) -> Path:
    """
    Build collision-safe destination path.

    If a file with the same name already exists in the destination,
    append an incrementing suffix to preserve all source files.
    """
    candidate = destination_dir / source_file.name
    if not candidate.exists():
        return candidate

    stem = source_file.stem
    suffix = source_file.suffix
    counter = 1
    while True:
        candidate = destination_dir / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


# ---------------------------------------------------------------------------
# GPU warm-up
# ---------------------------------------------------------------------------

def _warmup_gpu() -> None:
    """
    Absorb CUDA context and GPU library init before any timed measurement.

    Must be called once at the start of run_pipeline(), before the main
    processing loop. Without this, the first GPU file absorbs 3-10 s of
    library-initialization overhead, skewing that extension's average.
    """
    if torch is not None and torch.cuda.is_available():
        _dummy = torch.zeros(1, device="cuda")
        torch.cuda.synchronize()
        del _dummy

    if cudf is not None:
        _ = cudf.DataFrame({"_warmup": [0]})


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def run_timed(
    processor: Callable[[Path], None],
    file_path: Path,
) -> tuple[float | None, str | None]:
    """Time a single-file processor call; return elapsed seconds or error."""
    start = time.perf_counter()
    try:
        processor(file_path)
        return time.perf_counter() - start, None
    except Exception as exc:
        return None, str(exc)


def run_batch_timed(
    processor: Callable[[list[Path]], None],
    batch: list[Path],
) -> tuple[float | None, str | None]:
    """Time a batch processor call; return elapsed seconds or error string."""
    start = time.perf_counter()
    try:
        processor(batch)
        return time.perf_counter() - start, None
    except Exception as exc:
        return None, str(exc)


# ---------------------------------------------------------------------------
# CPU processing functions  (batch signatures)
# ---------------------------------------------------------------------------

def process_video_cpu(paths: list[Path]) -> None:
    """CPU baseline: decode up to 8 frames per video with OpenCV, resilient to corrupt files."""
    if cv2 is None:
        raise RuntimeError("opencv-python is not available")
    for path in paths:
        try:
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                continue
            for _ in range(8):
                ok, _ = cap.read()
                if not ok:
                    break
            cap.release()
        except Exception:
            pass


def process_audio_cpu(paths: list[Path]) -> None:
    """CPU baseline: load each waveform and compute a light statistic."""
    if torchaudio is None:
        raise RuntimeError("torchaudio is not available")
    for path in paths:
        try:
            waveform, _ = torchaudio.load(str(path))
            _ = float(waveform.abs().mean())
        except Exception:
            pass


def process_text_cpu(paths: list[Path]) -> None:
    """CPU baseline: parse JSON or count text length for each file with per-file error isolation."""
    for path in paths:
        try:
            if path.suffix.lower() == ".json":
                with path.open("r", encoding="utf-8", errors="ignore") as f:
                    _ = json.load(f)
            else:
                with path.open("r", encoding="utf-8", errors="ignore") as f:
                    _ = sum(len(line) for line in f)
        except Exception:
            pass


def _read_single_excel_cpu(path: Path) -> int:
    try:
        df = pd.read_excel(path, engine="openpyxl")
        return int(df.shape[0])
    except Exception:
        return 0


def process_xlsx_cpu(paths: list[Path]) -> None:
    """CPU baseline: read each spreadsheet with pandas/openpyxl in parallel."""
    with ThreadPoolExecutor(max_workers=min(8, len(paths) or 1)) as executor:
        list(executor.map(_read_single_excel_cpu, paths))


# ---------------------------------------------------------------------------
# GPU processing functions  (batch signatures)
# ---------------------------------------------------------------------------

def process_video_gpu_raw(paths: list[Path]) -> None:
    """
    Internal: build one DALI pipeline for a list of files and run it once.

    Do NOT call this directly from the processing loop — use
    process_video_gpu() which wraps this with binary-search failure isolation.

    Building the pipeline once per batch (not once per file) is the critical
    fix: pipeline.build() compiles CUDA kernels and allocates device memory,
    costing hundreds of ms to seconds when done per-file.
    """
    if Pipeline is None or fn is None or types is None:
        raise RuntimeError("nvidia-dali-cuda120 is not available")

    pipe = Pipeline(batch_size=len(paths), num_threads=2, device_id=0, seed=42)
    with pipe:
        video = fn.readers.video(
            device="gpu",
            filenames=[str(p) for p in paths],
            sequence_length=8,
            random_shuffle=False,
            shard_id=0,
            num_shards=1,
            normalized=False,
            image_type=types.RGB,
            dtype=types.UINT8,
        )
        pipe.set_outputs(fn.resize(video, resize_x=224, resize_y=224))
    pipe.build()
    pipe.run()


def _batch_dali_with_fallback(
    paths: list[Path],
) -> tuple[float, list[str]]:
    """
    Run DALI on a batch of video files with binary-search failure isolation.

    Algorithm
    ---------
    1. Try the full batch (zero overhead when all files are valid).
    2. On failure, split the batch in half and recurse on each half.
    3. A single-file batch that still fails identifies a definitively bad file.
    4. After recursion, all confirmed-good files are gathered and re-run as
       ONE batch so they still benefit from full batch-level GPU throughput.

    Cost of finding 1 bad file in 50
    ---------------------------------
    Binary search: ~log2(50) ≈ 6 extra pipeline builds.
    Sequential:    up to 50 extra pipeline builds.
    Counting whole batch as error: 0 builds but all 50 files lost.

    Returns
    -------
    (total_gpu_seconds_for_good_files, list_of_bad_file_path_strings)
    """
    elapsed, err = run_batch_timed(process_video_gpu_raw, paths)
    if err is None:
        return elapsed, []  # fast path — entire batch succeeded

    # Base case: a single-file batch failed → this file is definitively bad.
    if len(paths) == 1:
        return 0.0, [str(paths[0])]

    # Recurse on each half to isolate which files are bad.
    mid = len(paths) // 2
    left_elapsed, left_bad = _batch_dali_with_fallback(paths[:mid])
    right_elapsed, right_bad = _batch_dali_with_fallback(paths[mid:])

    bad_set = set(left_bad + right_bad)
    good_files = [p for p in paths if str(p) not in bad_set]

    if not good_files:
        return 0.0, list(bad_set)

    # Re-run all good files as a single batch for accurate combined timing.
    final_elapsed, final_err = run_batch_timed(process_video_gpu_raw, good_files)
    if final_err is None:
        return final_elapsed, list(bad_set)

    # Very unlikely: good-file re-batch also failed — fall back to summing
    # the leaf timings collected during the recursive descent.
    return left_elapsed + right_elapsed, list(bad_set)


def process_video_gpu(paths: list[Path]) -> None:
    """
    GPU path for MP4: DALI decode + resize with binary-search failure isolation.

    Good files in the batch always run at full batch speed. Bad files
    (corrupt/unsupported codecs) are identified by name and isolated.
    """
    elapsed, bad_files = _batch_dali_with_fallback(paths)
    for f in bad_files:
        print(f"GPU error [mp4] bad file isolated: {Path(f).name}")


def process_audio_gpu(paths: list[Path]) -> None:
    """
    GPU path for WAV: per-waveform GPU loop with a single CUDA sync at the end.

    WAV files in this dataset are 4-13 MB (avg 7.3 MB). Stacking 50 padded
    waveforms would need ~350 MB of GPU memory at peak, which risks OOM on
    smaller GPUs. Instead we loop per waveform but hold the cuda.synchronize()
    until all 50 have been submitted, so the GPU can overlap work across files.
    """
    if torch is None or torchaudio is None:
        raise RuntimeError("torch/torchaudio are not available")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available for torch")

    for path in paths:
        try:
            waveform, _ = torchaudio.load(str(path))
            waveform = waveform.cuda(non_blocking=True)
            torch.stft(waveform[0], n_fft=1024, return_complex=True)
        except Exception:
            pass

    # One synchronize for the entire batch — not one per file.
    torch.cuda.synchronize()


def process_text_gpu(paths: list[Path]) -> None:
    """
    GPU path for TXT/JSON: batch read into cuDF Series for vectorized GPU string analytics.

    Avoids fragile tabular JSON schemas while leveraging GPU for string length,
    tokenization, and character analytics across all batch files simultaneously.
    """
    if cudf is None:
        raise RuntimeError("cudf-cu12 is not available")

    raw_texts = []
    for path in paths:
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                raw_texts.append(f.read())
        except Exception:
            pass

    if raw_texts:
        s = cudf.Series(raw_texts)
        _ = int(s.str.len().sum())


def _read_excel_df_gpu(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_excel(path, engine="openpyxl")
    except Exception:
        return None


def process_xlsx_gpu(paths: list[Path]) -> None:
    """
    GPU path for XLSX: parallel CPU ingest + single batched cuDF conversion & operations.
    """
    if cudf is None:
        raise RuntimeError("cudf-cu12 is not available")

    with ThreadPoolExecutor(max_workers=min(8, len(paths) or 1)) as executor:
        results = list(executor.map(_read_excel_df_gpu, paths))

    frames = [df for df in results if df is not None and not df.empty]
    if frames:
        combined_pdf = pd.concat(frames, ignore_index=True)
        gdf = cudf.from_pandas(combined_pdf)
        _ = int(len(gdf))


# ---------------------------------------------------------------------------
# Processor registry
# ---------------------------------------------------------------------------

def get_processors_for_type(
    ext: str,
) -> tuple[Callable[[list[Path]], None], Callable[[list[Path]], None]]:
    """Resolve the CPU/GPU batch-processor pair for a file extension."""
    if ext == "mp4":
        return process_video_cpu, process_video_gpu
    if ext == "wav":
        return process_audio_cpu, process_audio_gpu
    if ext in {"txt", "json"}:
        return process_text_cpu, process_text_gpu
    if ext == "xlsx":
        return process_xlsx_cpu, process_xlsx_gpu
    raise ValueError(f"Unsupported extension: {ext}")


# ---------------------------------------------------------------------------
# File organization
# ---------------------------------------------------------------------------

def copy_to_organized(file_path: Path, extension: str) -> Path:
    """Copy a source file into the mapped organized subfolder."""
    destination_folder = OUTPUT_DIR / EXTENSION_TO_FOLDER[extension]
    destination = unique_destination_path(destination_folder, file_path)
    shutil.copy2(file_path, destination)
    return destination


# ---------------------------------------------------------------------------
# Benchmark DataFrame builder
# ---------------------------------------------------------------------------

def build_benchmark_dataframe(
    stats: dict[str, dict[str, float | int]],
) -> pd.DataFrame:
    """Convert accumulated benchmark stats to a CSV-ready DataFrame."""
    rows = []
    for ext in sorted(stats.keys()):
        record = stats[ext]
        file_count = int(record["files"]) if record["files"] else 0
        cpu_total = float(record["cpu_time"]) if record["cpu_time"] else 0.0
        gpu_total = float(record["gpu_time"]) if record["gpu_time"] else 0.0

        cpu_avg_ms = (cpu_total / file_count) * 1000 if file_count else 0.0
        gpu_avg_ms = (gpu_total / file_count) * 1000 if file_count else 0.0
        speedup = (cpu_total / gpu_total) if gpu_total > 0 else 0.0

        rows.append(
            {
                "file_type": ext,
                "num_files": file_count,
                "cpu_time_sec": cpu_total,
                "gpu_time_sec": gpu_total,
                "cpu_avg_ms": cpu_avg_ms,
                "gpu_avg_ms": gpu_avg_ms,
                "speedup_ratio": speedup,
                "cpu_errors": int(record["cpu_errors"]),
                "gpu_errors": int(record["gpu_errors"]),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline() -> pd.DataFrame:
    """
    Execute the full organization + benchmarking workflow.

    Returns
    -------
    Benchmark DataFrame for downstream usage or display.
    """
    ensure_output_dirs()
    grouped_files = discover_files(INPUT_DIR)

    stats: dict[str, dict[str, float | int]] = {
        ext: {
            "files": 0,
            "cpu_time": 0.0,
            "gpu_time": 0.0,
            "cpu_errors": 0,
            "gpu_errors": 0,
        }
        for ext in SUPPORTED_EXTENSIONS
    }

    total_supported = sum(len(files) for files in grouped_files.values())
    print(f"Discovered {total_supported} supported files in {INPUT_DIR}")

    # Warm up GPU runtimes before any timing begins.
    _warmup_gpu()

    for ext, files in grouped_files.items():
        cpu_processor, gpu_processor = get_processors_for_type(ext)
        total_files = len(files)

        for batch_start in range(0, total_files, BATCH_SIZE):
            batch = files[batch_start : batch_start + BATCH_SIZE]
            stats[ext]["files"] += len(batch)

            cpu_elapsed, cpu_err = run_batch_timed(cpu_processor, batch)
            if cpu_elapsed is not None:
                stats[ext]["cpu_time"] += cpu_elapsed
            else:
                stats[ext]["cpu_errors"] += len(batch)
                print(
                    f"CPU error [{ext}] batch starting at index "
                    f"{batch_start}: {cpu_err}"
                )

            gpu_elapsed, gpu_err = run_batch_timed(gpu_processor, batch)
            if gpu_elapsed is not None:
                stats[ext]["gpu_time"] += gpu_elapsed
            else:
                stats[ext]["gpu_errors"] += len(batch)
                print(
                    f"GPU error [{ext}] batch starting at index "
                    f"{batch_start}: {gpu_err}"
                )

            for path in batch:
                copy_to_organized(path, ext)

            # Progress feedback — same cadence as before (every ~250 files).
            processed_so_far = batch_start + len(batch)
            if processed_so_far % 250 < BATCH_SIZE:
                print(f"[{ext}] Processed {processed_so_far}/{total_files} files")

    benchmark_df = build_benchmark_dataframe(stats)
    benchmark_df.to_csv(BENCHMARK_CSV, index=False)

    print(f"Benchmark CSV written to: {BENCHMARK_CSV}")
    print("Summary:")
    if len(benchmark_df) > 0:
        print(benchmark_df.to_string(index=False))
    else:
        print("No supported files found.")

    return benchmark_df


if __name__ == "__main__":
    run_pipeline()
