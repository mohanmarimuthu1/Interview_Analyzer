"""Benchmark FacePipeline FPS on a video file or webcam.

Usage:
    python scripts/bench_face.py                     # webcam (index 0)
    python scripts/bench_face.py --video samples/test.mp4
    python scripts/bench_face.py --video samples/test.mp4 --frames 300
"""

import argparse
import platform
import time

import cv2

from interviewlens.vision.face import FacePipeline


def _hw_info() -> str:
    lines = [
        f"OS: {platform.system()} {platform.release()}",
        f"Python: {platform.python_version()}",
        f"CPU: {platform.processor() or 'unknown'}",
    ]
    try:
        import torch

        if torch.cuda.is_available():
            lines.append(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            lines.append("GPU: none (CPU-only)")
    except ImportError:
        lines.append("GPU: torch not installed")
    return "\n".join(lines)


def bench(video_source: int | str, max_frames: int) -> None:
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Cannot open source: {video_source!r}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_native = cap.get(cv2.CAP_PROP_FPS) or 0.0
    print(f"\nSource  : {video_source}")
    if total_frames > 0:
        print(f"Frames  : {total_frames}  native FPS: {fps_native:.1f}")

    print("\nHardware:")
    print(_hw_info())
    print()

    with FacePipeline(static_mode=False) as pipe:
        n_face = 0
        n_proc = 0
        t0 = time.perf_counter()

        while n_proc < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            result = pipe(frame)
            n_proc += 1
            if result is not None:
                n_face += 1

        elapsed = time.perf_counter() - t0

    cap.release()

    fps = n_proc / elapsed if elapsed > 0 else 0.0
    det_rate = 100.0 * n_face / n_proc if n_proc else 0.0
    print(f"Processed : {n_proc} frames in {elapsed:.2f}s")
    print(f"FPS       : {fps:.1f}")
    print(f"Detect    : {n_face}/{n_proc} ({det_rate:.0f}%)")

    target = 60
    ok = fps >= target
    mark = "PASS" if ok else "BELOW TARGET"
    print(f"\n[{mark}]  target >= {target} FPS on GPU  (measured on CPU here)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--video", default=None, help="path to video file (omit for webcam)")
    p.add_argument("--frames", type=int, default=200, help="max frames to process")
    args = p.parse_args()

    source: int | str = 0 if args.video is None else args.video
    bench(source, args.frames)


if __name__ == "__main__":
    main()
