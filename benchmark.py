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
    RESULTS_DIR,
    SUPPORTED_EXTENSIONS,
    _warmup_gpu,
    discover_files,
    get_processors_for_type,
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
) -> tuple[float, int]:
    """
    Run one batch processor against a file list in BATCH_SIZE chunks.

    Returns (total_time_seconds, total_error_count).  The benchmark
    continues through failures so one bad batch does not abort a long run.
    """
    total_time = 0.0
    errors = 0
    for batch_start in range(0, len(files), BATCH_SIZE):
        batch = files[batch_start : batch_start + BATCH_SIZE]
        elapsed, err = run_batch_timed(processor, batch)
        if elapsed is not None:
            total_time += elapsed
        else:
            errors += len(batch)
    return total_time, errors


def benchmark_per_type(grouped: dict[str, list[Path]]) -> pd.DataFrame:
    """Measure CPU/GPU total time for each supported file type."""
    rows = []
    for ext in sorted(SUPPORTED_EXTENSIONS):
        files = grouped.get(ext, [])
        if not files:
            continue

        cpu_proc, gpu_proc = get_processors_for_type(ext)
        cpu_time, cpu_errors = _timed_batch_run(cpu_proc, files)
        gpu_time, gpu_errors = _timed_batch_run(gpu_proc, files)
        speedup = (cpu_time / gpu_time) if gpu_time > 0 else math.nan

        rows.append(
            {
                "file_type": ext,
                "num_files": len(files),
                "cpu_time_sec": cpu_time,
                "gpu_time_sec": gpu_time,
                "speedup_ratio": speedup,
                "cpu_errors": cpu_errors,
                "gpu_errors": gpu_errors,
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
            cpu_t, _ = _timed_batch_run(cpu_proc, subset)
            gpu_t, _ = _timed_batch_run(gpu_proc, subset)
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
    Load benchmark.csv if it exists; otherwise run benchmark_per_type().

    Delete benchmark.csv to force a fresh timing run.
    """
    if BENCHMARK_CSV.exists():
        print(f"Loading existing benchmark from {BENCHMARK_CSV}")
        print("(Delete benchmark.csv to force a fresh timing run.)")
        return pd.read_csv(BENCHMARK_CSV)
    print("No existing benchmark.csv found — running full benchmark.")
    return benchmark_per_type(grouped)


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


if __name__ == "__main__":
    main()
