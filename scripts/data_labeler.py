#!/usr/bin/env python3
import os
import sys
import csv
import argparse
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}


class ImageLabeler(tk.Tk):
    def __init__(self, image_paths, csv_path):
        super().__init__()
        self.title("Image Labeler")
        self.configure(bg="white")

        default_font = ("Helvetica", 14)

        # High contrast button colors
        BUTTON_BG = "#e6e6e6"   # light grey
        BUTTON_FG = "black"

        self.option_add("*Font", default_font)
        self.option_add("*Button.Background", BUTTON_BG)
        self.option_add("*Button.Foreground", BUTTON_FG)
        self.option_add("*Label.Background", "white")
        self.option_add("*Label.Foreground", "black")
        self.option_add("*Entry.Background", "white")
        self.option_add("*Entry.Foreground", "black")
        self.option_add("*Entry.BorderWidth", 2)
        self.option_add("*Entry.Relief", "solid")

        self.image_paths = image_paths
        self.csv_path = csv_path
        self.current_idx = 0
        self.labels = {}  # image_path -> label string
        self.photo = None  # keep reference to PhotoImage

        # Load existing labels if CSV already exists
        if os.path.exists(self.csv_path):
            self._load_existing_labels()
        # Image display
        self.image_label = tk.Label(self, bg="white")
        self.image_label.pack(padx=10, pady=10)

        # Image path text
        self.path_label = tk.Label(self, text="", wraplength=800, fg="gray20")
        self.path_label.pack(padx=10, pady=(0, 10))

        # Label entry
        entry_frame = tk.Frame(self, bg="white")
        entry_frame.pack(padx=10, pady=(0, 10), fill="x")

        tk.Label(entry_frame, text="Label:", bg="white").pack(side="left")
        self.label_entry = tk.Entry(entry_frame, width=50)
        self.label_entry.pack(side="left", padx=5, fill="x", expand=True)

        # Buttons
        button_frame = tk.Frame(self, bg="white")
        button_frame.pack(padx=10, pady=10)

        self.prev_button = tk.Button(
            button_frame, text="Previous", bg=BUTTON_BG, fg=BUTTON_FG,
            command=self.prev_image
        )
        self.prev_button.pack(side="left", padx=5)

        self.next_button = tk.Button(
            button_frame, text="Next / Save", bg=BUTTON_BG, fg=BUTTON_FG,
            command=self.next_image
        )
        self.next_button.pack(side="left", padx=5)

        self.save_button = tk.Button(
            button_frame, text="Save CSV Now", bg=BUTTON_BG, fg=BUTTON_FG,
            command=self.save_csv_manual
        )
        self.save_button.pack(side="left", padx=5)

        self.quit_button = tk.Button(
            button_frame, text="Quit", bg=BUTTON_BG, fg=BUTTON_FG,
            command=self.on_quit
        )
        self.quit_button.pack(side="left", padx=5)

        # Status bar
        self.status_label = tk.Label(self, text="", anchor="w", bg="white", fg="black")
        self.status_label.pack(padx=10, pady=(0, 5), fill="x")

        # Bind Enter key to "Next / Save"
        self.bind("<Return>", lambda event: self.next_image())

        # If resuming, start at first unlabeled image
        self._jump_to_first_unlabeled()

        # Show first image
        self.update_display()

    def _load_existing_labels(self):
        """Load labels from an existing CSV file."""
        try:
            with open(self.csv_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    path = row.get("image_path", "")
                    label = row.get("label", "")
                    print(label)
                    if path in self.image_paths:
                        self.labels[path] = label
        except Exception as e:
            print(f"Warning: could not load existing labels from {self.csv_path}: {e}", file=sys.stderr)

    def _write_csv(self):
        """Write all labels to CSV."""
        try:
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["image_path", "label"])
                for path in self.image_paths:
                    label = self.labels.get(path, "")
                    writer.writerow([path, label])
        except Exception as e:
            print(f"Error writing CSV: {e}", file=sys.stderr)

    def save_csv_manual(self):
        """Manual save button callback."""
        self.save_current_label()  # ensure current label is stored
        messagebox.showinfo("Saved", f"Labels saved to:\n{self.csv_path}")

    def _jump_to_first_unlabeled(self):
        """When resuming, start at the first image without a non-empty label."""
        for i, path in enumerate(self.image_paths):
            label = self.labels.get(path, "")
            if label.strip() == "":
                self.current_idx = i
                return
        # If everything is labeled, just start at the last one
        self.current_idx = len(self.image_paths) - 1

    def update_display(self):
        """Update the displayed image, label entry, and status."""
        if not self.image_paths:
            self.path_label.config(text="No images found.")
            self.status_label.config(text="No images to label.")
            return

        path = self.image_paths[self.current_idx]
        self.path_label.config(text=path)

        # Load and resize image
        try:
            img = Image.open(path)
            max_width, max_height = 800, 600
            img.thumbnail((max_width, max_height))
            self.photo = ImageTk.PhotoImage(img)
            self.image_label.config(image=self.photo, text="")
        except Exception as e:
            self.image_label.config(text=f"Error loading image: {e}", image="")
            self.photo = None

        # Pre-fill label if exists
        existing_label = self.labels.get(path, "")
        self.label_entry.delete(0, tk.END)
        self.label_entry.insert(0, existing_label)
        self.label_entry.focus_set()

        total = len(self.image_paths)
        labeled_count = sum(1 for v in self.labels.values() if v.strip() != "")
        self.status_label.config(
            text=f"Image {self.current_idx + 1}/{total}   |   Labeled {labeled_count}/{total}"
        )

    def save_current_label(self):
        """Save the label for the current image (in memory and autosave CSV)."""
        if not self.image_paths:
            return
        path = self.image_paths[self.current_idx]
        label_text = self.label_entry.get().strip()
        self.labels[path] = label_text
        # Autosave after each label
        self._write_csv()

    def next_image(self):
        """Save current label and go to next image."""
        if not self.image_paths:
            return

        self.save_current_label()

        if self.current_idx < len(self.image_paths) - 1:
            self.current_idx += 1
            self.update_display()
        else:
            messagebox.showinfo("Done", "You have reached the last image.")
            self.update_display()

    def prev_image(self):
        """Go to previous image (no autosave here, only navigation)."""
        if not self.image_paths:
            return
        if self.current_idx > 0:
            self.current_idx -= 1
            self.update_display()
        else:
            messagebox.showinfo("Info", "This is the first image.")

    def on_quit(self):
        """Save current label and quit."""
        self.save_current_label()
        self.destroy()

def collect_images(image_dir):
    images = []
    for name in sorted(os.listdir(image_dir)):
        ext = os.path.splitext(name)[1].lower()
        if ext in SUPPORTED_EXTS:
            images.append(os.path.join(image_dir, name))
    return images


def main():
    parser = argparse.ArgumentParser(description="Simple image labeling GUI.")
    parser.add_argument("image_dir", help="Path to directory containing images.")
    parser.add_argument(
        "--csv",
        help="Path to CSV file for labels. If it exists, it will be loaded; otherwise created. "
             "Default: labels.csv inside image_dir",
        default=None,
    )
    args = parser.parse_args()

    image_dir = args.image_dir
    if not os.path.isdir(image_dir):
        print(f"Error: {image_dir} is not a directory or does not exist.")
        sys.exit(1)

    image_paths = collect_images(image_dir)
    if not image_paths:
        print(f"No supported images found in {image_dir}.")
        sys.exit(1)

    csv_path = args.csv or os.path.join(image_dir, "labels.csv")

    app = ImageLabeler(image_paths, csv_path)
    app.mainloop()


if __name__ == "__main__":
    main()
