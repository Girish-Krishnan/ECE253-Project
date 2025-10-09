#!/usr/bin/env python3
"""
GUI Video Frame Scrubber & Saver (with video name prefix + rotate 90° button)

Usage:
  python gui_frame_scrubber.py --video input.mov --outdir ./images

Controls:
  - Slider: drag to any frame
  - Frame box: type frame index and press Enter or click Go
  - Prev / Next buttons: step 1 frame
  - Save button (or Space): save current frame as JPG (named videoName_frame_xxxxxx.jpg)
  - Rotate 90° button: rotates display & saved frame clockwise by 90° each press
  - Left / Right arrow keys: step frames
"""

import argparse
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk

MAX_DISPLAY_W = 1280
MAX_DISPLAY_H = 720


def bgr_to_tk(frame_bgr):
    """Convert BGR (OpenCV) to Tkinter-compatible PhotoImage."""
    h, w = frame_bgr.shape[:2]
    scale = min(MAX_DISPLAY_W / w, MAX_DISPLAY_H / h, 1.0)
    if scale < 1.0:
        frame_bgr = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    return ImageTk.PhotoImage(img)


def rotate_image(frame, k=1):
    """Rotate frame clockwise by 90° * k."""
    return cv2.rotate(frame, [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE][(k - 1) % 3]) if k else frame


class VideoScrubberGUI:
    def __init__(self, root, video_path: Path, outdir: Path):
        self.root = root
        self.root.title("Video Frame Scrubber")
        self.video_path = video_path
        self.outdir = outdir
        self.outdir.mkdir(parents=True, exist_ok=True)

        # Open video
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            messagebox.showerror("Error", f"Could not open video: {self.video_path}")
            root.destroy()
            return

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 0.0
        if self.total_frames <= 0:
            self.total_frames = 1_000_000

        self.idx = 0
        self.rotation = 0  # 0, 1, 2, 3 for 0°, 90°, 180°, 270°
        ok, frame = self._read_frame(self.idx)
        if not ok:
            messagebox.showerror("Error", "Failed to read the first frame.")
            root.destroy()
            return
        self.current_frame = frame

        # GUI layout
        self.image_label = ttk.Label(self.root)
        self.image_label.grid(row=0, column=0, columnspan=7, sticky="nsew", padx=8, pady=8)

        # Slider
        self.slider = ttk.Scale(self.root, from_=0, to=self.total_frames - 1, orient="horizontal", command=self._on_slider)
        self.slider.grid(row=1, column=0, columnspan=7, sticky="ew", padx=8)

        # Frame number input
        ttk.Label(self.root, text="Frame:").grid(row=2, column=0, sticky="e", padx=(8, 2), pady=8)
        self.frame_var = tk.StringVar(value=str(self.idx))
        self.frame_entry = ttk.Entry(self.root, textvariable=self.frame_var, width=10)
        self.frame_entry.grid(row=2, column=1, sticky="w")
        self.frame_entry.bind("<Return>", self._on_go_frame)

        self.go_btn = ttk.Button(self.root, text="Go", command=self._on_go_frame)
        self.go_btn.grid(row=2, column=2, sticky="w")

        # Prev / Next
        self.prev_btn = ttk.Button(self.root, text="◀ Prev", command=self.prev_frame)
        self.prev_btn.grid(row=2, column=3, sticky="e", padx=4)
        self.next_btn = ttk.Button(self.root, text="Next ▶", command=self.next_frame)
        self.next_btn.grid(row=2, column=4, sticky="w", padx=4)

        # Save
        self.save_btn = ttk.Button(self.root, text="Save JPG", command=self.save_current_frame)
        self.save_btn.grid(row=2, column=5, sticky="e", padx=(4, 4))

        # Rotate button
        self.rotate_btn = ttk.Button(self.root, text="Rotate 90°", command=self.rotate_90)
        self.rotate_btn.grid(row=2, column=6, sticky="e", padx=(4, 8))

        # Status bar
        self.status_var = tk.StringVar()
        self.status = ttk.Label(self.root, textvariable=self.status_var, anchor="w")
        self.status.grid(row=3, column=0, columnspan=7, sticky="ew", padx=8, pady=(0, 8))

        for c in range(7):
            self.root.grid_columnconfigure(c, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        # Keybindings
        self.root.bind("<Left>", lambda e: self.prev_frame())
        self.root.bind("<Right>", lambda e: self.next_frame())
        self.root.bind("<space>", lambda e: self.save_current_frame())

        self._update_display()

    def _read_frame(self, idx):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = self.cap.read()
        return ok, frame

    def _set_index(self, new_idx):
        new_idx = max(0, min(int(new_idx), self.total_frames - 1))
        if new_idx == self.idx:
            return
        ok, frame = self._read_frame(new_idx)
        if ok and frame is not None:
            self.idx = new_idx
            self.current_frame = frame
            self._update_display()

    def _update_display(self):
        frame = self.current_frame
        if self.rotation:
            frame = self._rotate_frame(frame)
        tkimg = bgr_to_tk(frame)
        self.image_label.img_ref = tkimg
        self.image_label.configure(image=tkimg)
        self.slider.configure(to=self.total_frames - 1)
        self.slider.set(self.idx)
        self.frame_var.set(str(self.idx))
        t = (self.idx / self.fps) if self.fps > 0 else 0.0
        total_disp = "?" if self.total_frames >= 1_000_000 else f"{self.total_frames - 1}"
        rotation_text = f" | Rot: {self.rotation * 90}°" if self.rotation else ""
        self.status_var.set(
            f"{self.video_path.name}{rotation_text} | frame {self.idx} / {total_disp} | t={t:0.3f}s | FPS={self.fps:.3f}"
        )

    def _rotate_frame(self, frame):
        if self.rotation == 1:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotation == 2:
            return cv2.rotate(frame, cv2.ROTATE_180)
        elif self.rotation == 3:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    def _on_slider(self, val):
        try:
            target = int(float(val))
        except ValueError:
            return
        if target != self.idx:
            ok, frame = self._read_frame(target)
            if ok and frame is not None:
                self.idx = target
                self.current_frame = frame
                self._update_display()

    def _on_go_frame(self, event=None):
        try:
            target = int(self.frame_var.get())
        except ValueError:
            messagebox.showwarning("Invalid input", "Please enter a valid integer frame index.")
            return
        self._set_index(target)

    def prev_frame(self):
        self._set_index(self.idx - 1)

    def next_frame(self):
        self._set_index(self.idx + 1)

    def rotate_90(self):
        self.rotation = (self.rotation + 1) % 4
        self._update_display()

    def save_current_frame(self):
        base = self.video_path.stem
        frame_to_save = self._rotate_frame(self.current_frame)
        filename = self.outdir / f"{base}_frame_{self.idx:06d}.jpg"
        ok = cv2.imwrite(str(filename), frame_to_save)
        if ok:
            self.status_var.set(f"Saved: {filename.name}")
        else:
            self.status_var.set("Save failed!")

    def close(self):
        if self.cap:
            self.cap.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", "-v", required=True, help="Path to input video (.MOV, .MP4, etc.)")
    parser.add_argument("--outdir", "-o", default="./images", help="Directory to save JPG frames")
    args = parser.parse_args()

    video_path = Path(args.video)
    outdir = Path(args.outdir)

    root = tk.Tk()
    app = VideoScrubberGUI(root, video_path, outdir)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.close(), root.destroy()))
    root.mainloop()


if __name__ == "__main__":
    main()
