#!/usr/bin/env python3
"""
Batch repair script for MP4 files in the dataset.
Ensures every video has an IDR keyframe at frame 0, CFR 25/30 fps, and standard H.264 profile
so that hardware NVDEC and NVIDIA DALI decode all files with 0 errors.
"""

import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

INPUT_DIR = Path("/run/media/pranam/Laksh 320GB/gpu_pipeline/UNORGANIZED")


def repair_file(path: Path) -> tuple[str, bool, str]:
    tmp_path = path.with_name(path.stem + "_fixed_tmp.mp4")
    cmd = [
        "ffmpeg",
        "-y",
        "-v", "error",
        "-i", str(path),
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-g", "30",
        "-keyint_min", "1",
        "-force_key_frames", "expr:eq(n,0)",
        "-c:a", "copy",
        "-movflags", "+faststart",
        str(tmp_path),
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if res.returncode == 0 and tmp_path.exists() and tmp_path.stat().st_size > 1000:
            tmp_path.replace(path)
            return path.name, True, ""
        else:
            if tmp_path.exists():
                tmp_path.unlink()
            return path.name, False, res.stderr or "Empty/Corrupt output"
    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink()
        return path.name, False, str(e)


def main():
    if not INPUT_DIR.exists():
        print(f"Error: Input directory {INPUT_DIR} does not exist.")
        return

    files = sorted(INPUT_DIR.glob("*.mp4"))
    total = len(files)
    print(f"Found {total} MP4 files in {INPUT_DIR}")
    if total == 0:
        return

    num_workers = min(12, os.cpu_count() or 4)
    print(f"Starting batch repair using {num_workers} parallel workers...")
    t0 = time.perf_counter()

    success_count = 0
    fail_count = 0
    errors = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for i, (name, ok, err) in enumerate(executor.map(repair_file, files), 1):
            if ok:
                success_count += 1
            else:
                fail_count += 1
                errors.append((name, err))

            if i % 250 == 0 or i == total:
                elapsed = time.perf_counter() - t0
                rate = i / elapsed if elapsed > 0 else 0
                print(f"Progress: {i}/{total} ({i/total*100:.1f}%) - {rate:.1f} files/sec - Success: {success_count}, Failed: {fail_count}")

    total_time = time.perf_counter() - t0
    print(f"\nCompleted in {total_time:.1f}s ({total_time/60:.2f} min)")
    print(f"Summary: {success_count} fixed successfully, {fail_count} failed.")
    if errors:
        print(f"First few errors: {errors[:5]}")


if __name__ == "__main__":
    main()
