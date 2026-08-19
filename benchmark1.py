#!/usr/bin/env python3
"""
Standalone benchmark runner for CPU vs GPU multimodal processing.

This script reuses processors from main.py and generates publication-ready plots:
1) Bar chart: CPU vs GPU total time per file type
2) Line chart: Overall speedup ratio vs number of files benchmarked

All outputs are written to /media/kart/Laksh 320GB/PIPELINE/results/.

Re-processing behaviour
-----------------------
If benchmark.csv already exists (written by main.py), this script loads it
directly and skips re-processing the full dataset.  Delete benchmark.csv
first if you want a fresh timing run.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from main import (
    BATCH_SIZE,
    BENCHMARK_CSV,
    INPUT_DIR,
    MP4_GPU_ERROR_LOG,
    NVDEC_SUPPORTED_CODECS,
    RESULTS_DIR,
    SUPPORTED_EXTENSIONS,
    _DALI_VIDEO_PROCS,
    _MP4_FILE_TO_CODEC,
    _warmup_gpu,
    _validate_and_group_mp4,
    _prevalidate_mp4_decodable,
    discover_files,
    get_processors_for_type,
    init_dali_video_processors,
    run_batch_timed,
    warm_xlsx_parquet_cache,
)

BAR_CHART_PATH = RESULTS_DIR / "cpu_vs_gpu_time_per_filetype.png"
LINE_CHART_PATH = RESULTS_DIR / "speedup_vs_number_of_files.png"
SPEEDUP_TABLE_PATH = RESULTS_DIR / "speedup_table.csv"


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------

def _timed_batch_run(
    processor,
    files: list[Path],
    ext: str = "",
) -> tuple[float, int, int]:
    """
    Run one batch processor against a file list in BATCH_SIZE chunks.

    Returns (total_time_seconds, total_error_count, total_batches).
    total_batches is required alongside total_error_count so callers can
    compute a batch-level error RATE (errors / batches). Dividing
    errors (a batch-level count) by file_count instead is a units
    mismatch that hides near-total failure at large BATCH_SIZE - see
    benchmark_per_type() below.

    FIX: a failed batch is counted as 1 error (one failed batch-call),
    not len(batch) errors.  The previous len(batch) accounting inflated
    GPU error rates — e.g. one bad file in a 50-file batch was reported
    as 50 errors, which pushed the gpu_error_rate over the 5% threshold
    and incorrectly marked entire file-types as UNRELIABLE.

    FIX 2: when ext=="mp4" and a GPU batch fails, the failing filenames
    are appended to MP4_GPU_ERROR_LOG so errors are diagnosable without
    re-running the full benchmark.
    """
    total_time = 0.0
    errors = 0
    batches = 0
    for batch_start in range(0, len(files), BATCH_SIZE):
        batch = files[batch_start : batch_start + BATCH_SIZE]
        batches += 1
        elapsed, err = run_batch_timed(processor, batch)
        if elapsed is not None:
            total_time += elapsed
        else:
            errors += 1  # 1 failed batch, not len(batch) inflated file-count
            if ext == "mp4" and err is not None:
                RESULTS_DIR.mkdir(parents=True, exist_ok=True)
                with MP4_GPU_ERROR_LOG.open("a") as _log:
                    for _p in batch:
                        _log.write(f"{_p}\n")
    return total_time, errors, batches


def benchmark_per_type(grouped: dict[str, list[Path]]) -> pd.DataFrame:
    """Measure CPU/GPU total time for each supported file type.

    Includes data_quality and gpu_pipeline_fails columns (matching the schema
    produced by main.py's build_benchmark_dataframe) so that
    save_publication_plots() can correctly exclude unreliable rows.
    """
    rows = []
    for ext in sorted(SUPPORTED_EXTENSIONS):
        files = grouped.get(ext, [])
        if not files:
            continue

        cpu_proc, gpu_proc = get_processors_for_type(ext)
        cpu_time, cpu_errors, _cpu_batches = _timed_batch_run(cpu_proc, files)
        gpu_time, gpu_errors, gpu_batches = _timed_batch_run(gpu_proc, files, ext=ext)

        file_count = len(files)
        # FIX: rate must match the granularity of the count. gpu_errors is
        # a count of FAILED BATCHES, so the rate is errors/batches, not
        # errors/file_count - the old errors/file_count version could
        # report a 100%-failing GPU path as "OK" once BATCH_SIZE made the
        # batch count much smaller than the file count.
        gpu_err_rate = gpu_errors / gpu_batches if gpu_batches else 0.0
        if gpu_time > 0 and gpu_err_rate < 0.05:
            speedup = cpu_time / gpu_time
            quality = "OK"
        else:
            speedup = math.nan
            quality = f"UNRELIABLE (err_rate={gpu_err_rate:.1%})"

        rows.append(
            {
                "file_type": ext,
                "num_files": file_count,
                "cpu_time_sec": cpu_time,
                "gpu_time_sec": gpu_time,
                "cpu_avg_ms": (cpu_time / file_count) * 1000 if file_count else 0.0,
                "gpu_avg_ms": (gpu_time / file_count) * 1000 if file_count else 0.0,
                "speedup_ratio": speedup,
                "cpu_errors": cpu_errors,
                "gpu_errors": gpu_errors,
                "gpu_pipeline_fails": 0,   # pre-flight already excluded failures
                "data_quality": quality,
            }
        )

    return pd.DataFrame(rows)


def benchmark_speedup_curve(grouped: dict[str, list[Path]]) -> pd.DataFrame:
    """
    Measure speedup as dataset size increases.

    We sample the first N files per file type and aggregate total CPU/GPU times
    across all modalities for each N.  Files within each checkpoint are still
    processed in BATCH_SIZE chunks so GPU amortization applies at every point.
    """
    max_files_any_type = max((len(v) for v in grouped.values()), default=0)
    if max_files_any_type == 0:
        return pd.DataFrame(
            columns=["num_files", "cpu_time_sec", "gpu_time_sec", "speedup_ratio"]
        )

    checkpoints = [10, 50, 100, 250, 500, 1000, 2000, 5000]
    checkpoints = [n for n in checkpoints if n <= max_files_any_type]
    if max_files_any_type not in checkpoints:
        checkpoints.append(max_files_any_type)

    rows = []
    for n in checkpoints:
        cpu_total = 0.0
        gpu_total = 0.0

        for ext, files in grouped.items():
            subset = files[: min(n, len(files))]
            if not subset:
                continue
            cpu_proc, gpu_proc = get_processors_for_type(ext)
            cpu_t, _, _ = _timed_batch_run(cpu_proc, subset)
            gpu_t, _, _ = _timed_batch_run(gpu_proc, subset)
            cpu_total += cpu_t
            gpu_total += gpu_t

        speedup = (cpu_total / gpu_total) if gpu_total > 0 else math.nan
        rows.append(
            {
                "num_files": n,
                "cpu_time_sec": cpu_total,
                "gpu_time_sec": gpu_total,
                "speedup_ratio": speedup,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CSV loader (skip re-processing when data already exists)
# ---------------------------------------------------------------------------

def load_or_run_per_type(grouped: dict[str, list[Path]]) -> pd.DataFrame:
    """
    Always run a fresh benchmark against the provided (pre-filtered) grouped
    file list and save results to benchmark.csv.

    NOTE: The old behaviour of loading a cached benchmark.csv was removed
    because it caused the pre-flight MP4 exclusion logic (which correctly
    narrows grouped["mp4"] before this call) to be silently bypassed —
    the stale CSV was returned unchanged, making the fairness fix a no-op.
    Delete benchmark.csv manually if you want to inspect old results.
    """
    print("Running full benchmark against filtered file set.")
    df = benchmark_per_type(grouped)
    BENCHMARK_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(BENCHMARK_CSV, index=False)
    print(f"Fresh benchmark written to {BENCHMARK_CSV}")
    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_publication_plots(
    per_type_df: pd.DataFrame,
    curve_df: pd.DataFrame,
) -> None:
    """Create and save publication-ready matplotlib figures.

    Rows with data_quality != 'OK' are excluded from the bar chart so that
    unreliable speedup numbers (e.g. MP4 with high GPU error rate, XLSX with
    first-run conversion bias) do not appear in published figures.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Use a clean, high-contrast style suitable for papers.
    plt.style.use("seaborn-v0_8-whitegrid")

    # Filter to trustworthy rows only.
    if "data_quality" in per_type_df.columns:
        plot_df = per_type_df[per_type_df["data_quality"] == "OK"].copy()
        excluded = per_type_df[
            per_type_df["data_quality"] != "OK"
        ]["file_type"].tolist()
        if excluded:
            print(f"Bar chart: excluding unreliable file types: {excluded}")
    else:
        plot_df = per_type_df.copy()

    # Plot 1: grouped bar chart for CPU vs GPU times.
    fig, ax = plt.subplots(figsize=(11, 6), dpi=200)
    x = list(range(len(plot_df)))
    width = 0.38

    ax.bar(
        [i - width / 2 for i in x],
        plot_df["cpu_time_sec"],
        width,
        label="CPU Time",
        color="#4c72b0",
    )
    ax.bar(
        [i + width / 2 for i in x],
        plot_df["gpu_time_sec"],
        width,
        label="GPU Time",
        color="#dd8452",
    )

    ax.set_title("CPU vs GPU Runtime per File Type", fontsize=15, fontweight="bold")
    ax.set_xlabel("File Type", fontsize=12)
    ax.set_ylabel("Total Time (seconds)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["file_type"].tolist(), fontsize=11)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(BAR_CHART_PATH)
    plt.close(fig)

    # Plot 2: speedup as workload size grows.
    fig, ax = plt.subplots(figsize=(11, 6), dpi=200)
    ax.plot(
        curve_df["num_files"],
        curve_df["speedup_ratio"],
        marker="o",
        linewidth=2.5,
        color="#55a868",
        label="Speedup (CPU/GPU)",
    )
    ax.set_title("GPU Speedup vs Number of Files", fontsize=15, fontweight="bold")
    ax.set_xlabel("Files per Type (N)", fontsize=12)
    ax.set_ylabel("Speedup Ratio", fontsize=12)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(LINE_CHART_PATH)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run all benchmarking steps and emit tables/plots to disk."""
    grouped = discover_files(INPUT_DIR)

    _warmup_gpu()  # absorb CUDA/cuDF init before any timing

    # -----------------------------------------------------------------------
    # FIX: initialise DALI video pipelines before any MP4 timing.
    #
    # When benchmark.py is run standalone (not via run_pipeline() in main.py)
    # the module-level globals _DALI_VIDEO_PROCS and _MP4_FILE_TO_CODEC are
    # empty, so every MP4 GPU batch raises RuntimeError and counts as an
    # error.  This made the entire speedup curve invalid whenever MP4 files
    # were present.  We mirror the same init sequence used in run_pipeline().
    # -----------------------------------------------------------------------
    if "mp4" in grouped and BATCH_SIZE > 0:
        print(f"Pre-validating {len(grouped['mp4'])} MP4 files (codec detection) …")
        codec_groups, bad_mp4 = _validate_and_group_mp4(grouped["mp4"])
        for bad in bad_mp4:
            print(f"Pre-validation failed [mp4]: {bad.name}")

        # ------------------------------------------------------------------
        # FAIRNESS: exclude unsupported-codec files from BOTH CPU and GPU.
        # CPU and GPU must always operate on the exact same input set.
        # ------------------------------------------------------------------
        unsupported_codecs = [
            c for c in codec_groups if c not in NVDEC_SUPPORTED_CODECS
        ]
        for codec in unsupported_codecs:
            n = len(codec_groups[codec])
            print(
                f"  ⚠ Excluding codec='{codec}' ({n} files) from BOTH CPU and GPU "
                f"timing — not supported by NVDEC (fair exclusion)."
            )
            del codec_groups[codec]

        # Only build pipelines for the remaining supported codecs.
        supported_groups = {
            c: f for c, f in codec_groups.items()
            if c in NVDEC_SUPPORTED_CODECS
        }
        if supported_groups:
            init_dali_video_processors(supported_groups)

        # Candidate set: files whose codec pipeline built successfully.
        # Iterate codec_groups (already filtered to supported codecs above),
        # NOT the original grouped["mp4"] which still contains all paths.
        candidate_mp4 = [
            p
            for files in codec_groups.values()
            for p in files
            if _MP4_FILE_TO_CODEC.get(str(p)) in _DALI_VIDEO_PROCS
        ]

        # ------------------------------------------------------------------
        # FAIRNESS: frame-level decodability pre-check (symmetric).
        # Files that OpenCV cannot decode are excluded from BOTH paths.
        # ------------------------------------------------------------------
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
        grouped["mp4"] = decodable_mp4

    # XLSX: ensure Parquet cache is warmed before any timing begins.
    # Without this, process_xlsx_gpu() raises RuntimeError on missing cache.
    if "xlsx" in grouped:
        xlsx_files = grouped["xlsx"]
        print(f"Pre-warming XLSX Parquet cache ({len(xlsx_files)} files) …")
        n_converted = warm_xlsx_parquet_cache(xlsx_files)
        print(
            f"  → {n_converted} new conversions, "
            f"{len(xlsx_files) - n_converted} cache hits."
        )

    per_type_df = load_or_run_per_type(grouped)
    curve_df = benchmark_speedup_curve(grouped)

    if per_type_df.empty:
        print("No supported files found in dataset. Nothing to benchmark.")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_publication_plots(per_type_df, curve_df)

    # Persist table so paper workflow can directly use CSV artifacts.
    per_type_df.to_csv(SPEEDUP_TABLE_PATH, index=False)

    print("Final speedup table:")
    print(per_type_df.to_string(index=False))
    print(f"Saved: {SPEEDUP_TABLE_PATH}")
    print(f"Saved: {BAR_CHART_PATH}")
    print(f"Saved: {LINE_CHART_PATH}")

    # FIX: emit a clean, filtered publication CSV instead of a raw-data copy.
    # speedup_table.csv now contains only OK-quality rows with a readable
    # column order and values rounded to 2 d.p.  Raw data stays in benchmark.csv.
    pub_cols = ["file_type", "num_files", "cpu_avg_ms", "gpu_avg_ms", "speedup_ratio"]
    pub_df = (
        per_type_df[per_type_df["data_quality"] == "OK"][pub_cols]
        .round({"cpu_avg_ms": 2, "gpu_avg_ms": 2, "speedup_ratio": 2})
        .reset_index(drop=True)
    )
    pub_df.to_csv(SPEEDUP_TABLE_PATH, index=False)
    print(f"Publication table ({len(pub_df)} OK rows) written to {SPEEDUP_TABLE_PATH}")


if __name__ == "__main__":
    main()
