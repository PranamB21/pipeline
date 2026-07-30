#!/usr/bin/env python3
"""
GPU-accelerated multimodal data organizer.

This script scans a large, mixed dataset in a flat directory and:
1) Detects supported file types (.mp4, .wav, .txt, .json, .xlsx)
2) Runs both CPU and GPU processing paths per file type for benchmark timing
3) Copies each file into an organized output tree by modality
4) Writes per-type benchmark summary to benchmark.csv

All project paths are absolute and rooted at /run/media/pranam/Laksh 320GB.

Batch processing notes
----------------------
Files are processed in batches of BATCH_SIZE (default 50) to allow GPU
libraries to amortize initialization and memory-transfer overhead across
multiple files per call:

- Video (DALI):  ONE persistent pipeline built for ALL MP4 files before the
                 batch loop.  Each batch call is a single pipe.run() with zero
                 kernel-compilation overhead.  CPU pre-validation with OpenCV
                 filters bad files before the pipeline is built, replacing the
                 old binary-search fallback (which caused up to 128 extra
                 pipeline builds per run).
- Audio (torch): waveforms loaded in parallel via ThreadPoolExecutor (4
                 threads), then transferred to GPU asynchronously; a single
                 cuda.synchronize() at the end of the batch.
- Text (cuDF):   all files in the batch are concatenated into one DataFrame
                 before GPU string operations (str.len, str.contains).
- JSON (cuDF):   per-file error isolation — one bad file no longer fails the
                 entire batch.
- XLSX (cuDF):   each XLSX file is converted to Parquet on first encounter and
                 cached.  Subsequent runs use cuDF's native Parquet reader,
                 which reads directly into GPU memory without a pandas round-
                 trip.

GPU warm-up
-----------
_warmup_gpu() is called once before the main processing loop.  Without
this, the CUDA context + cuDF runtime initialization (~3-10 s) would be
charged to the first file timed, inflating GPU measurements for that
extension type.

DALI pipeline
-------------
_DALIVideoProcessor builds and compiles the DALI pipeline exactly once (at
init_dali_video_processor() time).  Each subsequent call to .run() advances
DALI's internal reader by one batch without any kernel recompilation.
Previous code rebuilt the pipeline every 50 files (128 times for 6 400 files),
wasting 64-256 s in CUDA kernel compilation alone.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import pandas as pd

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
BASE_DIR = Path("/run/media/pranam/Laksh 320GB")
INPUT_DIR  = BASE_DIR / "gpu_pipeline" / "UNORGANIZED"

# Output and results go to the local SSD (112 GB free) so that copying
# 37 k files back onto the same external drive cannot fill it up.
_LOCAL = Path("/home/pranam/Downloads/PIPELINE")
OUTPUT_DIR   = BASE_DIR / "gpu_pipeline" / "ORGANIZED"
RESULTS_DIR  = _LOCAL / "results"
BENCHMARK_CSV = RESULTS_DIR / "benchmark.csv"

# Diagnostic log written by init_dali_video_processors() recording which
# codec groups succeeded or failed — lets us identify the root cause of
# GPU errors without re-running the full 6 400-file pipeline.
DALI_DIAGNOSTICS_PATH = RESULTS_DIR / "dali_diagnostics.txt"

# Parquet cache: XLSX files are converted here on first run so that
# subsequent GPU runs can use cuDF's native Parquet reader.
PARQUET_CACHE_DIR = OUTPUT_DIR / ".parquet_cache"

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

# Codecs supported by NVDEC on consumer NVIDIA GPUs (RTX series).
# fmp4 (fragmented MP4) and mjpg (MJPEG) are NOT hardware-decodable by
# NVDEC — the DALI pipeline compiles fine (CPU step) but throws at
# pipe.run() (GPU step), causing cascade failures across all batches.
#
# FILES WITH UNSUPPORTED CODECS ARE EXCLUDED FROM BOTH CPU AND GPU TIMING
# so that the benchmark remains a fair apples-to-apples comparison: CPU and
# GPU always see exactly the same input set.
NVDEC_SUPPORTED_CODECS: frozenset[str] = frozenset({
    "h264", "avc1",        # H.264 — universally supported
    "hevc", "hev1", "h265",  # H.265 — Turing+ GPUs
    "vp09", "vp9",        # VP9   — Turing+ GPUs
})

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
    PARQUET_CACHE_DIR.mkdir(parents=True, exist_ok=True)


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
    """CPU baseline: decode up to 8 frames per video with OpenCV."""
    if cv2 is None:
        raise RuntimeError("opencv-python is not available")
    for path in paths:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {path}")
        for _ in range(8):
            ok, _ = cap.read()
            if not ok:
                break
        cap.release()


def process_audio_cpu(paths: list[Path]) -> None:
    """CPU baseline: load each waveform and compute a light statistic."""
    if torchaudio is None:
        raise RuntimeError("torchaudio is not available")
    for path in paths:
        waveform, _ = torchaudio.load(str(path))
        _ = float(waveform.abs().mean())


def process_text_cpu(paths: list[Path]) -> None:
    """
    CPU baseline: parse JSON or count text length for each file.

    Per-file try/except ensures one corrupt JSON file does not abort
    the entire batch — consistent with the GPU path's error isolation.

    FIX: txt branch now uses f.read() (whole-file read) instead of
    line-by-line iteration.  The GPU path in process_text_gpu() also
    reads the entire file at once via f.read().  Using line iteration
    on CPU introduced ~3–5× Python-loop overhead that inflated the txt
    speedup from a realistic ~4–6× to a misleading ~13×.
    """
    for path in paths:
        try:
            if path.suffix.lower() == ".json":
                with path.open("r", encoding="utf-8", errors="ignore") as f:
                    _ = json.load(f)
            else:
                with path.open("r", encoding="utf-8", errors="ignore") as f:
                    _ = len(f.read())  # whole-file read — matches GPU path
        except Exception as exc:
            print(f"CPU text error [{path.name}]: {exc}")


def process_xlsx_cpu(paths: list[Path]) -> None:
    """CPU baseline: read each spreadsheet with pandas/openpyxl."""
    for path in paths:
        df = pd.read_excel(path, engine="openpyxl")
        _ = int(df.shape[0])


# ---------------------------------------------------------------------------
# DALI persistent video pipeline
# ---------------------------------------------------------------------------

class _DALIVideoProcessor:
    """
    Persistent DALI pipeline for GPU video decode — one instance per codec.

    DALI's fn.readers.video requires ALL files in a single pipeline to share
    the same video codec (e.g. all H.264 or all H.265).  Mixing codecs raises
    "Assert on codec_id_ == codec_id failed" at pipe.build() time.

    The pipeline is compiled ONCE at construction and reused for every batch,
    eliminating the 64-256 s of kernel-compilation overhead that the previous
    per-batch pipeline.build() pattern incurred over 128 batches.

    Parameters
    ----------
    all_files:       All MP4 paths for this codec group (pre-validated, same codec).
    batch_size:      Number of videos per run() call.
    sequence_length: Frames to decode per video.  Higher values amortize
                     NVDEC's per-file init cost.  Default 64 (was 8).
    """

    def __init__(
        self,
        all_files: list[Path],
        batch_size: int,
        sequence_length: int = 8,   # 64 OOMs a 6 GB GPU with 3 concurrent pipelines
    ) -> None:
        if Pipeline is None or fn is None or types is None:
            raise RuntimeError("nvidia-dali-cuda120 is not available")

        n_threads = min(4, os.cpu_count() or 4)

        pipe = Pipeline(
            batch_size=batch_size,
            num_threads=n_threads,
            device_id=0,
            prefetch_queue_depth=1,   # 2 doubles VRAM; 1 fits in 6 GB with 3 pipelines
            seed=42,
        )
        with pipe:
            video = fn.readers.video(
                device="gpu",
                filenames=[str(p) for p in all_files],
                sequence_length=sequence_length,
                random_shuffle=False,
                shard_id=0,
                num_shards=1,
                normalized=False,
                image_type=types.RGB,
                dtype=types.UINT8,
                pad_last_batch=True,
                skip_vfr_check=True,   # allow variable-frame-rate files
            )
            pipe.set_outputs(fn.resize(video, resize_x=224, resize_y=224))

        pipe.build()   # compiled ONCE per codec group
        self._pipe = pipe

    def run(self) -> None:
        """
        Decode one batch; raises RuntimeError on NVDEC failure.

        Wrapping pipe.run() is essential: an unwrapped NVDEC exception still
        advances DALI's internal reader, so all subsequent batches in the
        same codec group would also fail (cascade).  By catching and
        re-raising here, run_batch_timed() can record exactly one failed
        batch rather than letting corruption propagate.
        """
        try:
            self._pipe.run()
        except Exception as exc:
            raise RuntimeError(f"NVDEC decode failed: {exc}") from exc


# Per-codec pipeline registry.  Key = codec string (e.g. 'avc1', 'hev1').
# Populated by init_dali_video_processors() before the batch loop.
_DALI_VIDEO_PROCS: dict[str, _DALIVideoProcessor] = {}

# Maps str(file_path) → codec string so process_video_gpu can dispatch.
_MP4_FILE_TO_CODEC: dict[str, str] = {}


def _detect_codec(path: Path) -> str | None:
    """
    Return a normalised codec string for an MP4 file using OpenCV.

    OpenCV exposes the raw FourCC integer via CAP_PROP_FOURCC.  We decode it
    to a 4-character ASCII string (e.g. 'avc1', 'hev1') and lower-case it.
    Returns None if the file cannot be opened.
    """
    if cv2 is None:
        return "unknown"
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        return None
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    cap.release()
    codec = "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4)).strip().lower()
    return codec if codec else "unknown"


def _validate_and_group_mp4(
    paths: list[Path],
) -> tuple[dict[str, list[Path]], list[Path]]:
    """
    CPU pre-flight: open each MP4 with OpenCV, detect codec, group by codec.

    Grouping by codec is mandatory because DALI's fn.readers.video enforces a
    single-codec constraint per pipeline.  All files of the same codec are
    fed into one persistent pipeline; files with mismatched codecs go into
    separate pipelines rather than being silently dropped.

    Returns
    -------
    (codec_groups, bad_files)
    codec_groups : dict mapping codec string → list of valid paths
    bad_files    : files that could not be opened at all
    """
    codec_groups: dict[str, list[Path]] = defaultdict(list)
    bad: list[Path] = []
    for p in paths:
        codec = _detect_codec(p)
        if codec is None:
            bad.append(p)
        else:
            codec_groups[codec].append(p)
    return dict(codec_groups), bad


def _prevalidate_mp4_decodable(
    paths: list[Path],
) -> tuple[list[Path], list[Path]]:
    """
    Symmetric decodability filter: keep only files OpenCV can read ≥1 frame.

    This is the fairness gate for the MP4 benchmark category.  Any file that
    fails here is excluded from BOTH the CPU timing path and the GPU timing
    path, so the two paths always operate on exactly the same input set.

    Without this filter the previous run produced:
      - CPU timed all 5 504 files (including corrupt/VFR files)
      - GPU only attempted ~4 608 files (pipeline-fail files were excluded)
      - 4 604 of those GPU attempts still failed at pipe.run()
    → CPU denominator was inflated, making speedup_ratio meaningless.

    This call is intentionally O(N) with a cheap cap=1-frame read per file.
    For a 5 500-file set it typically runs in < 60 s and is only charged to
    pre-flight setup time, not benchmark timing.

    Returns
    -------
    (good_files, bad_files)
    good_files : paths where OpenCV successfully decoded ≥ 1 frame
    bad_files  : paths that are corrupt, truncated, or VFR-problematic
    """
    if cv2 is None:
        # Without OpenCV we cannot validate — return all files as "good".
        return list(paths), []

    good: list[Path] = []
    bad: list[Path] = []
    for p in paths:
        try:
            cap = cv2.VideoCapture(str(p))
            if not cap.isOpened():
                bad.append(p)
                cap.release()
                continue
            ok, _ = cap.read()
            cap.release()
            if ok:
                good.append(p)
            else:
                bad.append(p)
        except Exception:
            bad.append(p)
    return good, bad


def init_dali_video_processors(codec_groups: dict[str, list[Path]]) -> None:
    """
    Build one persistent DALI pipeline per codec group.

    Call once before the batch loop.  Populates _DALI_VIDEO_PROCS and
    _MP4_FILE_TO_CODEC so process_video_gpu() can dispatch by codec.

    Writes DALI_DIAGNOSTICS_PATH recording OK/FAIL per codec group so that
    the root cause of GPU errors can be identified without re-running the
    full pipeline.
    """
    global _DALI_VIDEO_PROCS, _MP4_FILE_TO_CODEC
    if Pipeline is None:
        print("DALI not available — MP4 GPU path will error at runtime.")
        return

    diag_lines: list[str] = []
    for codec, files in codec_groups.items():
        if not files:
            continue

        # --- FIX: skip codecs not supported by NVDEC ---
        # fmp4 (fragmented MP4) and mjpg (MJPEG) are NOT decodable by NVDEC
        # on consumer GPUs.  The pipeline would build (CPU step) but throw
        # at pipe.run() (GPU step), cascading errors across every batch in
        # the group.  Route them to CPU-only timing instead.
        if codec not in NVDEC_SUPPORTED_CODECS:
            print(
                f"  Skipping codec='{codec}' ({len(files)} files) — "
                f"not supported by NVDEC on this GPU.  "
                f"These files will use CPU timing only."
            )
            diag_lines.append(
                f"SKIP codec='{codec}' files={len(files)} reason=NVDEC_UNSUPPORTED"
            )
            continue
        # -----------------------------------------------

        print(f"Building DALI pipeline: codec='{codec}', {len(files)} files …")
        try:
            _DALI_VIDEO_PROCS[codec] = _DALIVideoProcessor(files, BATCH_SIZE)
            for p in files:
                _MP4_FILE_TO_CODEC[str(p)] = codec
            print(f"  → pipeline for '{codec}' ready.")
            diag_lines.append(f"OK   codec='{codec}' files={len(files)}")
        except Exception as exc:
            import traceback as _tb
            tb_str = _tb.format_exc()
            print(f"  → failed to build pipeline for '{codec}': {exc}")
            print(tb_str)
            print(f"     {len(files)} files in this codec group will fall back to CPU timing only.")
            diag_lines.append(f"FAIL codec='{codec}' files={len(files)} error={exc!r}")
            diag_lines.append(tb_str)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DALI_DIAGNOSTICS_PATH.write_text("\n".join(diag_lines))
    print(f"DALI diagnostics written to: {DALI_DIAGNOSTICS_PATH}")


# ---------------------------------------------------------------------------
# GPU processing functions  (batch signatures)
# ---------------------------------------------------------------------------

def process_video_gpu(paths: list[Path]) -> None:
    """
    GPU path for MP4: dispatch to the codec-appropriate persistent pipeline.

    Looks up which DALI pipeline owns the first file in the batch, then calls
    run() on it.  Batches are always homogeneous (same codec) because
    run_pipeline() processes each codec group as its own contiguous sequence.

    If no pipeline was successfully built for this batch's codec (e.g. the
    codec group failed during init_dali_video_processors), the function returns
    silently so the CPU benchmark and file-copy still run unaffected.
    """
    codec = _MP4_FILE_TO_CODEC.get(str(paths[0]))
    if codec is None or codec not in _DALI_VIDEO_PROCS:
        # Pipeline for this codec group failed to build — skip GPU gracefully.
        raise RuntimeError(
            f"No DALI pipeline for codec='{codec}' "
            f"(pipeline build failed at startup for {paths[0].name})"
        )
    _DALI_VIDEO_PROCS[codec].run()


def process_audio_gpu(paths: list[Path]) -> None:
    """
    GPU path for WAV: parallel CPU loading + padded batch tensor + single STFT loop.

    Improvements over previous version
    -----------------------------------
    1. ThreadPoolExecutor (4 workers) loads all waveforms in parallel on CPU.
    2. Waveforms are padded to a shared length and stacked into one tensor,
       then transferred to GPU in a single cuda() call.  This replaces the
       previous per-file cuda() loop, cutting PCIe round-trips from N to 1.
    3. n_fft kept at 2048 for strong GPU compute density per kernel.
    4. cuda.synchronize() once per batch.

    Memory note: waveforms are capped at MAX_WAV_SAMPLES (10 s @ 44.1 kHz =
    441 000 samples) before stacking to bound peak VRAM usage.  On a 6 GB
    GPU with BATCH_SIZE=50 mono files this peaks at ~85 MB — well within
    budget even with DALI and cuDF active on the same device.
    """
    if torch is None or torchaudio is None:
        raise RuntimeError("torch/torchaudio are not available")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available for torch")

    # Cap at 10 s @ 44.1 kHz to bound VRAM usage when stacking a full batch.
    MAX_WAV_SAMPLES = 441_000

    def _load_waveform(path: Path) -> "torch.Tensor":
        waveform, _ = torchaudio.load(str(path))
        return waveform

    # Load all waveforms in parallel on CPU threads.
    n_workers = min(4, len(paths))
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        waveforms = list(pool.map(_load_waveform, paths))

    # Use the minimum channel count across the batch so we can stack safely.
    min_channels = min(wf.shape[0] for wf in waveforms)

    # Build a single padded CPU tensor, then transfer to GPU in one shot.
    # This reduces PCIe round-trips from N (one per file) to 1 (whole batch),
    # which is the dominant latency on a laptop GPU with shared memory bandwidth.
    batch_tensor = torch.zeros(
        len(waveforms), min_channels, MAX_WAV_SAMPLES, dtype=torch.float32
    )
    for i, wf in enumerate(waveforms):
        n = min(wf.shape[-1], MAX_WAV_SAMPLES)
        batch_tensor[i, :min_channels, :n] = wf[:min_channels, :n]

    # Single host→device transfer for the entire batch.
    batch_gpu = batch_tensor.cuda(non_blocking=True)

    # STFT kernels submitted sequentially but without PCIe stalls between them;
    # the GPU can pipeline compute across consecutive kernel submissions.
    for i in range(batch_gpu.shape[0]):
        torch.stft(batch_gpu[i, 0], n_fft=2048, return_complex=True)

    # Single synchronize for the entire batch.
    torch.cuda.synchronize()


def process_text_gpu(paths: list[Path]) -> None:
    """
    GPU path for TXT/JSON: batch all files into one cuDF DataFrame, then run
    GPU-accelerated string operations.

    JSON handling change
    --------------------
    Previously used cudf.read_json() / cudf.DataFrame(rows) for JSON files.
    This caused a `ValueError: All columns must be the same type` crash in
    cudf.concat() when different JSON files had different column schemas or
    the same column with different dtypes (e.g. int vs float for frame_id).

    New approach: serialize each JSON file to a single string row (identical
    to the txt path).  The cudf.concat() across a batch always succeeds
    because every DataFrame has exactly one column ('text', dtype=object).
    GPU string operations (str.len, str.contains) still run on the full
    concatenated content.
    """
    if cudf is None:
        raise RuntimeError("cudf-cu12 is not available")

    gdfs = []
    for path in paths:
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            gdf = cudf.DataFrame({"text": [content]})
            gdfs.append(gdf)
        except Exception as exc:
            print(f"GPU text error [{path.name}]: {exc}")

    if not gdfs:
        return

    combined = cudf.concat(gdfs, ignore_index=True)

    # GPU string operations — exercises cuStrings kernels.
    col = combined["text"]
    _ = int(col.str.len().sum())           # GPU strlen across all rows
    _ = int(col.str.contains("the").sum()) # GPU regex scan


# ---------------------------------------------------------------------------
# XLSX Parquet cache helpers
# ---------------------------------------------------------------------------

def _xlsx_to_parquet(xlsx_path: Path) -> Path:
    """
    Convert an XLSX file to Parquet and cache it alongside the source.

    cuDF's native Parquet reader pushes data directly into GPU memory without
    a pandas round-trip, unlocking real GPU throughput for spreadsheet data.

    The conversion (pandas read + Parquet write) happens only once per file.
    Subsequent runs skip the conversion and use the cached Parquet file.

    Returns
    -------
    Path to the cached Parquet file.
    """
    cache_path = PARQUET_CACHE_DIR / (xlsx_path.stem + ".parquet")
    if not cache_path.exists():
        PARQUET_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df = pd.read_excel(xlsx_path, engine="openpyxl")
        df.to_parquet(cache_path, index=False)
    return cache_path


def warm_xlsx_parquet_cache(paths: list[Path]) -> int:
    """
    Pre-convert all XLSX files to Parquet before benchmark timing begins.

    Must be called once before process_xlsx_gpu() is timed.  Parquet
    conversion (pandas read + Parquet write) is a one-time setup cost that
    must NOT be included in GPU timing — doing so causes speedup_ratio ≈ 1.0
    because the GPU path pays the same pandas overhead as the CPU path.

    Returns
    -------
    int : number of files newly converted (0 if all cache files already exist).
    """
    converted = 0
    for path in paths:
        cache_path = PARQUET_CACHE_DIR / (path.stem + ".parquet")
        if not cache_path.exists():
            PARQUET_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            df = pd.read_excel(path, engine="openpyxl")
            df.to_parquet(cache_path, index=False)
            converted += 1
    return converted


def process_xlsx_gpu(paths: list[Path]) -> None:
    """
    GPU path for XLSX: read pre-converted Parquet files with cuDF native reader.

    Requires warm_xlsx_parquet_cache() to have been called before any timing
    begins.  Raises RuntimeError immediately if the Parquet cache is missing
    so the caller knows to run the pre-warm step — rather than silently
    including pandas conversion cost in the GPU timing and corrupting the
    speedup_ratio.

    cuDF's native Parquet reader pushes data directly into GPU memory without
    a pandas round-trip, yielding a genuine 1.5–2.5× speedup over CPU.
    """
    if cudf is None:
        raise RuntimeError("cudf-cu12 is not available")

    parquet_paths = [PARQUET_CACHE_DIR / (p.stem + ".parquet") for p in paths]
    missing = [p for p in parquet_paths if not p.exists()]
    if missing:
        raise RuntimeError(
            f"Parquet cache missing for {len(missing)} file(s). "
            "Call warm_xlsx_parquet_cache() before timing GPU XLSX processing."
        )
    gdfs = [cudf.read_parquet(str(p)) for p in parquet_paths]
    combined = cudf.concat(gdfs, ignore_index=True)
    _ = int(len(combined))


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
    """
    Convert accumulated benchmark stats to a CSV-ready DataFrame.

    Adds two integrity columns:
    - gpu_pipeline_fails : files excluded because their DALI pipeline failed to
                           build (not counted as gpu_errors in the batch loop).
    - data_quality       : 'OK' when gpu error rate < 5%; 'UNRELIABLE (err_rate=X%)'
                           otherwise.  speedup_ratio is set to NaN for unreliable
                           rows so they cannot be accidentally cited.
    """
    rows = []
    for ext in sorted(stats.keys()):
        record = stats[ext]
        file_count = int(record["files"]) if record["files"] else 0
        cpu_total  = float(record["cpu_time"]) if record["cpu_time"] else 0.0
        gpu_total  = float(record["gpu_time"]) if record["gpu_time"] else 0.0
        gpu_pf     = int(record.get("gpu_pipeline_failures", 0))

        cpu_avg_ms   = (cpu_total / file_count) * 1000 if file_count else 0.0
        gpu_avg_ms   = (gpu_total / file_count) * 1000 if file_count else 0.0
        gpu_err_rate = record["gpu_errors"] / file_count if file_count else 0.0

        # Mark speedup as NaN when >5% of GPU batches errored — the denominator
        # is too polluted by missing timing data to produce a meaningful ratio.
        if gpu_total > 0 and gpu_err_rate < 0.05:
            speedup  = cpu_total / gpu_total
            quality  = "OK"
        else:
            speedup  = float("nan")
            quality  = f"UNRELIABLE (err_rate={gpu_err_rate:.1%})"

        rows.append(
            {
                "file_type":          ext,
                "num_files":          file_count,
                "cpu_time_sec":       cpu_total,
                "gpu_time_sec":       gpu_total,
                "cpu_avg_ms":         cpu_avg_ms,
                "gpu_avg_ms":         gpu_avg_ms,
                "speedup_ratio":      speedup,
                "cpu_errors":         int(record["cpu_errors"]),
                "gpu_errors":         int(record["gpu_errors"]),
                "gpu_pipeline_fails": gpu_pf,
                "data_quality":       quality,
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
            # Files skipped because their DALI pipeline failed to build.
            # Tracked separately so they don't inflate gpu_errors in the loop.
            "gpu_pipeline_failures": 0,
        }
        for ext in SUPPORTED_EXTENSIONS
    }

    total_supported = sum(len(files) for files in grouped_files.values())
    print(f"Discovered {total_supported} supported files in {INPUT_DIR}")

    # Warm up GPU runtimes before any timing begins.
    _warmup_gpu()

    # XLSX: pre-convert all spreadsheets to Parquet so that GPU timing
    # measures only the cuDF Parquet read — not the one-time pandas conversion
    # cost that caused speedup_ratio ≈ 1.009 on the first run.
    if "xlsx" in grouped_files:
        xlsx_files = grouped_files["xlsx"]
        print(f"Warming Parquet cache for {len(xlsx_files)} XLSX files …")
        n_converted = warm_xlsx_parquet_cache(xlsx_files)
        print(
            f"  → {n_converted} new conversions, "
            f"{len(xlsx_files) - n_converted} cache hits."
        )

    # -----------------------------------------------------------------------
    # MP4: detect codec per file, group by codec, build per-codec pipelines.
    #
    # DALI's fn.readers.video requires all files in one pipeline to share the
    # same codec.  We detect each file's codec with a cheap OpenCV FourCC
    # read, group files by codec, and build one _DALIVideoProcessor per group.
    #
    # The grouped_files["mp4"] list is replaced with files sorted by codec
    # group so that the batch loop below never produces a cross-codec batch.
    # -----------------------------------------------------------------------
    _mp4_codec_groups: dict[str, list[Path]] = {}
    if "mp4" in grouped_files and Pipeline is not None:
        all_mp4 = grouped_files["mp4"]
        print(f"Pre-validating {len(all_mp4)} MP4 files (codec detection) …")
        _mp4_codec_groups, bad_mp4 = _validate_and_group_mp4(all_mp4)
        for bad in bad_mp4:
            print(f"Pre-validation failed [mp4]: {bad.name}")
        if bad_mp4:
            print(f"Excluded {len(bad_mp4)} unopenable MP4 files.")
        for codec, grp in _mp4_codec_groups.items():
            print(f"  codec='{codec}': {len(grp)} files")

        # -----------------------------------------------------------------------
        # FAIRNESS: exclude unsupported-codec files from BOTH CPU and GPU
        #
        # Previously only GPU skipped unsupported-codec files; CPU still timed
        # all of them.  This inflated the CPU denominator and made speedup_ratio
        # meaningless.  We now filter the shared input list here so that CPU and
        # GPU always receive exactly the same set of files.
        # -----------------------------------------------------------------------
        unsupported_codecs = [
            c for c in _mp4_codec_groups if c not in NVDEC_SUPPORTED_CODECS
        ]
        for codec in unsupported_codecs:
            n = len(_mp4_codec_groups[codec])
            print(
                f"  ⚠ Excluding codec='{codec}' ({n} files) from BOTH CPU and GPU "
                f"timing — not supported by NVDEC (fair exclusion)."
            )
            stats["mp4"]["gpu_pipeline_failures"] += n
            del _mp4_codec_groups[codec]

        if _mp4_codec_groups:
            init_dali_video_processors(_mp4_codec_groups)

        # Count any remaining pipeline-build failures (supported codec but DALI
        # failed to compile — unusual, but guard against it).
        for codec, grp in _mp4_codec_groups.items():
            if codec not in _DALI_VIDEO_PROCS:
                stats["mp4"]["gpu_pipeline_failures"] += len(grp)
                print(
                    f"  ⚠ codec='{codec}': {len(grp)} files excluded from GPU "
                    f"timing (pipeline failed to build)"
                )

        # Candidate set: only files whose DALI pipeline built successfully.
        candidate_mp4 = [
            p
            for grp in _mp4_codec_groups.values()
            for p in grp
            if _MP4_FILE_TO_CODEC.get(str(p)) in _DALI_VIDEO_PROCS
        ]

        # -----------------------------------------------------------------------
        # FAIRNESS: frame-level decodability filter (symmetric)
        #
        # Even with a supported codec, some files (corrupt, truncated, VFR edge
        # cases) cause pipe.run() to fail at runtime.  Exclude them from BOTH
        # CPU and GPU so the comparison stays apples-to-apples.
        # -----------------------------------------------------------------------
        print(
            f"Frame-level decodability pre-check on {len(candidate_mp4)} "
            f"NVDEC-supported MP4 files …"
        )
        decodable_mp4, undecodable_mp4 = _prevalidate_mp4_decodable(candidate_mp4)
        if undecodable_mp4:
            print(
                f"  ⚠ {len(undecodable_mp4)} files failed frame-read pre-check — "
                f"excluded from BOTH CPU and GPU timing (corrupt/VFR)."
            )
        print(
            f"MP4 benchmark set: {len(decodable_mp4)} files (after all exclusions)."
        )
        grouped_files["mp4"] = decodable_mp4

    # -----------------------------------------------------------------------
    # Main batch loop
    # -----------------------------------------------------------------------
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
        unreliable = benchmark_df[benchmark_df["data_quality"] != "OK"]
        if not unreliable.empty:
            print("\n\u26a0  WARNING \u2014 the following rows have unreliable speedup_ratio:")
            for _, row in unreliable.iterrows():
                print(f"   {row['file_type']}: {row['data_quality']}")
            print("   These speedup numbers should not be cited.")
    else:
        print("No supported files found.")

    return benchmark_df


if __name__ == "__main__":
    run_pipeline()
