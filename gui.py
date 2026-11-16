"""
gui.py - Tkinter app for Historical Letter Enhancement.
Run from the project folder with:  python gui.py

Open an image -> choose the algorithms -> press Process -> compare before/after.
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
from PIL import Image, ImageTk

from src.utils import load_image, save_image
from src.pipeline import (run_pipeline, save_steps, ENHANCE_OPTIONS, DENOISE_OPTIONS,
                          THRESHOLD_OPTIONS, CLEANUP_OPTIONS)

VIEW_W, VIEW_H = 520, 680          # size of each picture box


class App:
    def __init__(self, root):
        self.root = root
        root.title("Historical Letter Enhancement")

        self.rgb = None             # the loaded colour image
        self.result = None          # the processed black & white image
        self.steps = None           # every intermediate stage (dict)
        self._stage_photos = []     # keep references for the stages window
        self.rotate = 0             # 0 / 90 / 180 / 270
        self.photo_in = None        # keep references so images don't vanish
        self.photo_out = None

        self._build_controls()
        self._build_views()

    # ---------------------------------------------------------- layout
    def _build_controls(self):
        panel = ttk.Frame(self.root, padding=10)
        panel.grid(row=0, column=0, sticky="ns")

        ttk.Button(panel, text="Open Image...", command=self.open_image).pack(fill="x")
        ttk.Button(panel, text="Rotate 90°", command=self.rotate_image).pack(fill="x", pady=(4, 10))

        self.enhance = self._dropdown(panel, "Enhancement", ENHANCE_OPTIONS, "CLAHE")
        self.denoise = self._dropdown(panel, "Noise removal", DENOISE_OPTIONS, "Median Filter")
        self.threshold = self._dropdown(panel, "Binarization", THRESHOLD_OPTIONS, "Adaptive")
        self.cleanup = self._dropdown(panel, "Cleanup (morphology)", CLEANUP_OPTIONS,
                                      "Remove Specks + Clear Border")

        self.crop_on = tk.BooleanVar(value=True)
        ttk.Checkbutton(panel, text="Crop to paper", variable=self.crop_on).pack(anchor="w", pady=(8, 0))
        self.margin = self._slider(panel, "Crop margin", 0.0, 0.08, 0.005, 0.03)

        ttk.Separator(panel).pack(fill="x", pady=8)
        self.C = self._slider(panel, "Adaptive C (higher = fewer specks)", 0, 50, 1, 25)
        self.window = self._slider(panel, "Adaptive window size", 11, 101, 2, 51)
        self.global_t = self._slider(panel, "Global threshold", 0, 255, 1, 127)
        self.min_size = self._slider(panel, "Min speck size", 0, 60, 1, 15)

        ttk.Button(panel, text="Process", command=self.process).pack(fill="x", pady=(10, 4))
        ttk.Button(panel, text="Save Result...", command=self.save_result).pack(fill="x")
        ttk.Button(panel, text="Show All Stages", command=self.show_stages).pack(fill="x", pady=(10, 4))
        ttk.Button(panel, text="Save All Stages...", command=self.save_all_stages).pack(fill="x")

        self.status = ttk.Label(panel, text="Open an image to start.", wraplength=220)
        self.status.pack(anchor="w", pady=8)

    def _dropdown(self, parent, label, options, default):
        ttk.Label(parent, text=label).pack(anchor="w", pady=(6, 0))
        var = tk.StringVar(value=default)
        ttk.Combobox(parent, textvariable=var, values=options, state="readonly",
                     width=28).pack(fill="x")
        return var

    def _slider(self, parent, label, low, high, step, default):
        ttk.Label(parent, text=label).pack(anchor="w", pady=(4, 0))
        var = tk.DoubleVar(value=default)
        tk.Scale(parent, from_=low, to=high, resolution=step, orient="horizontal",
                 variable=var, length=220).pack(fill="x")
        return var

    def _build_views(self):
        views = ttk.Frame(self.root, padding=10)
        views.grid(row=0, column=1)

        ttk.Label(views, text="Original").grid(row=0, column=0)
        ttk.Label(views, text="Enhanced result").grid(row=0, column=1)

        # before an image is shown the label size is in text units, so keep it modest
        self.view_in = tk.Label(views, width=60, height=40, bg="#dddddd")
        self.view_out = tk.Label(views, width=60, height=40, bg="#dddddd")
        self.view_in.grid(row=1, column=0, padx=5)
        self.view_out.grid(row=1, column=1, padx=5)

    # --------------------------------------------------------- actions
    def open_image(self):
        path = filedialog.askopenfilename(
            title="Choose a letter image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")])
        if not path:
            return
        try:
            self.rgb = load_image(path, max_width=1600)
        except Exception as e:
            messagebox.showerror("Error", f"Could not open image:\n{e}")
            return
        self.rotate = 0
        self.result = None
        self.status.config(text=f"Loaded {os.path.basename(path)}  {self.rgb.shape[1]}x{self.rgb.shape[0]}")
        self._show(self.rgb, self.view_in, "in")
        self.view_out.config(image="", width=60, height=40)

    def rotate_image(self):
        if self.rgb is None:
            return
        self.rotate = (self.rotate + 90) % 360
        self._show(np.rot90(self.rgb, self.rotate // 90), self.view_in, "in")
        self.status.config(text=f"Rotated {self.rotate}°. Press Process.")

    def process(self):
        if self.rgb is None:
            messagebox.showinfo("No image", "Open an image first.")
            return
        self.status.config(text="Processing... (a few seconds)")
        self.root.update_idletasks()
        try:
            self.result, self.steps = run_pipeline(
                self.rgb, rotate=self.rotate, crop=self.crop_on.get(),
                crop_margin=self.margin.get(), enhance=self.enhance.get(),
                denoise=self.denoise.get(), threshold=self.threshold.get(),
                cleanup=self.cleanup.get(), C=self.C.get(), window=self.window.get(),
                global_t=int(self.global_t.get()), min_size=self.min_size.get())
        except Exception as e:
            messagebox.showerror("Error", f"Processing failed:\n{e}")
            self.status.config(text="Failed.")
            return
        self._show(self.result, self.view_out, "out")
        self.status.config(text="Done.")

    def save_result(self):
        if self.result is None:
            messagebox.showinfo("Nothing to save", "Press Process first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if path:
            save_image(self.result, path)
            self.status.config(text=f"Saved {os.path.basename(path)}")

    def show_stages(self):
        """Open a window showing the result of every processing stage."""
        if self.steps is None:
            messagebox.showinfo("Nothing yet", "Press Process first.")
            return
        win = tk.Toplevel(self.root)
        win.title("Intermediate outputs - every stage")
        self._stage_photos = []
        for col, (name, arr) in enumerate(self.steps.items()):
            ttk.Label(win, text=f"{col + 1}. {name}").grid(row=0, column=col, padx=4, pady=4)
            img = Image.fromarray(np.ascontiguousarray(arr))
            img.thumbnail((300, 420))
            photo = ImageTk.PhotoImage(img)
            self._stage_photos.append(photo)
            tk.Label(win, image=photo).grid(row=1, column=col, padx=4, pady=4)

    def save_all_stages(self):
        """Save every stage as a numbered PNG in a folder you choose."""
        if self.steps is None:
            messagebox.showinfo("Nothing yet", "Press Process first.")
            return
        folder = filedialog.askdirectory(title="Choose a folder for the stage images")
        if folder:
            save_steps(self.steps, folder)
            self.status.config(text=f"Saved {len(self.steps)} stage images to {folder}")

    # ------------------------------------------------------- helpers
    def _show(self, arr, label, which):
        """Shrink an array to fit the picture box and show it."""
        img = Image.fromarray(np.ascontiguousarray(arr))
        img.thumbnail((VIEW_W, VIEW_H))
        photo = ImageTk.PhotoImage(img)
        label.config(image=photo, width=photo.width(), height=photo.height())
        if which == "in":
            self.photo_in = photo
        else:
            self.photo_out = photo


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()