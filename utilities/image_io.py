"""Shared image batch/CLI scaffolding.

Every close-up detector (and the connector) needs the same plumbing: the list of
image extensions it accepts, a way to batch-process a folder, and a tiny CLI that
takes either a folder (batch) or a single image. That code used to be copied into
each detector file; it lives here once now.

Each detector supplies its OWN `process_image(path)` (the per-class print format
differs and is worth keeping per detector); these helpers drive it.
"""

import os
import sys


# Image file extensions the detectors accept.
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def list_images(folder):
    """Return sorted image filenames (not full paths) in `folder`."""
    return sorted(
        name for name in os.listdir(folder)
        if name.lower().endswith(IMAGE_EXTS)
    )


def process_folder(folder, process_image_fn):
    """Run `process_image_fn(path)` over every image file in `folder`."""
    names = list_images(folder)
    if not names:
        print(f"No image files found in {folder}")
        return
    print(f"Processing {len(names)} image(s) in {folder}")
    for name in names:
        process_image_fn(os.path.join(folder, name))


def run_cli(process_image_fn, default="input_images"):
    """Tiny CLI: a folder arg batch-processes it, else process the single image.

    Pass the caller's own `process_image(path)`. With no argument, falls back to
    the `default` target (the project's input folder).
    """
    target = sys.argv[1] if len(sys.argv) > 1 else default
    if os.path.isdir(target):
        process_folder(target, process_image_fn)
    else:
        process_image_fn(target)
