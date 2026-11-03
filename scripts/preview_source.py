"""Play a config's frame source in a window, with its depth map alongside if it has one.

Uses tkinter (bundled with Python), so it needs no extra installs. No pipeline runs:
this shows the raw source frames.

    .\\.venv\\Scripts\\python.exe scripts\\preview_source.py
    .\\.venv\\Scripts\\python.exe scripts\\preview_source.py --config configs/stub.yaml --scale 2

Close the window or press Esc to stop. The stub source only produces depth when its config
sets ``frame_source.options.depth: true``.
"""

from __future__ import annotations

import argparse
import time
import tkinter as tk
from pathlib import Path

import numpy as np

from lucid_director.core.config import load_config
from lucid_director.core.interfaces import FrameSource
from lucid_director.core.registry import registry
from lucid_director.core.types import Frame


def depth_to_rgb(depth: np.ndarray) -> np.ndarray:
    """Greyscale (H, W, 3) uint8 view of a depth map: near = dark, far = bright."""
    span = float(depth.max() - depth.min())
    normalized = (depth - depth.min()) / span if span > 0 else np.zeros_like(depth)
    grey = (normalized * 255).round().astype(np.uint8)
    return np.repeat(grey[..., None], 3, axis=2)


def compose(frame: Frame, scale: int) -> np.ndarray:
    """The frame image, plus its depth map to the right if present, enlarged by ``scale``."""
    panels = [frame.image]
    if frame.depth is not None:
        panels.append(depth_to_rgb(frame.depth))
    image = np.concatenate(panels, axis=1)
    # Nearest-neighbour enlargement: each pixel becomes a scale × scale block.
    return image.repeat(scale, axis=0).repeat(scale, axis=1)


def to_ppm(image: np.ndarray) -> bytes:
    """Encode an RGB uint8 image as binary PPM, a format tkinter can load directly."""
    height, width = image.shape[:2]
    return f"P6 {width} {height} 255\n".encode("ascii") + image.tobytes()


class Preview:
    def __init__(self, source: FrameSource, scale: int, fps: float) -> None:
        self.source = source
        self.scale = scale
        self.interval_s = 1.0 / fps
        self.shown = 0
        self.started = time.perf_counter()
        self.last_shown = self.started

        self.root = tk.Tk()
        self.root.bind("<Escape>", lambda _event: self.root.destroy())
        self.label = tk.Label(self.root)
        self.label.pack()
        self.photo: tk.PhotoImage | None = None  # kept referenced so tkinter doesn't discard it

    def run(self) -> None:
        self.root.after(0, self.step)
        self.root.mainloop()
        # Measured up to the last frame, not window close, which may be much later.
        elapsed = self.last_shown - self.started
        fps = self.shown / elapsed if elapsed > 0 else 0.0
        print(f"shown {self.shown} frames in {elapsed:.1f} s ({fps:.1f} fps)")

    def step(self) -> None:
        tick = time.perf_counter()
        frame = self.source.next_frame()
        if frame is None:
            self.root.title(f"{self.root.title()}  (end)")
            return

        self.photo = tk.PhotoImage(data=to_ppm(compose(frame, self.scale)), format="PPM")
        self.label.configure(image=self.photo)
        self.shown += 1
        self.last_shown = time.perf_counter()
        self.root.title(f"frame {frame.index}  t={frame.timestamp - self.started:.2f}s")

        # Wait out the rest of this frame's time slot, so playback runs at the target fps.
        spent_s = time.perf_counter() - tick
        delay_ms = max(1, round((self.interval_s - spent_s) * 1000))
        self.root.after(delay_ms, self.step)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config", type=Path, default=Path("configs/stub.yaml"), help="(default: %(default)s)"
    )
    parser.add_argument("--scale", type=int, default=3, help="enlargement factor (default: 3)")
    parser.add_argument("--fps", type=float, default=30.0, help="playback rate (default: 30)")
    args = parser.parse_args(argv)

    section = load_config(args.config).frame_source
    with registry.create("frame_source", section.type, section) as source:
        print(f"{section.type} source, {section.resolution[0]}x{section.resolution[1]}")
        print(f"provides: {sorted(source.provides()) or 'nothing extra'}")
        print(f"scene: {source.scene_summary()}")
        Preview(source, args.scale, args.fps).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
