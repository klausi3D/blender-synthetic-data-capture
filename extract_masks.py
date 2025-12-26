#!/usr/bin/env python3
"""
Extract masks from RGBA images for 3DGS training.

For images with transparent background (alpha channel), this script:
1. Extracts the alpha channel as a mask
2. Creates a white background version for training (optional)

Usage:
    python extract_masks.py /path/to/capture/images
    python extract_masks.py /path/to/capture --create-white-bg
"""

import os
import sys
import argparse
from pathlib import Path

try:
    from PIL import Image
    import numpy as np
except ImportError:
    print("Please install Pillow and numpy: pip install Pillow numpy")
    sys.exit(1)


def extract_masks_from_folder(images_folder: Path, output_masks_folder: Path = None,
                               create_white_bg: bool = False, white_bg_folder: Path = None):
    """Extract masks from RGBA images in a folder."""

    if output_masks_folder is None:
        output_masks_folder = images_folder.parent / "masks"

    output_masks_folder.mkdir(parents=True, exist_ok=True)

    if create_white_bg:
        if white_bg_folder is None:
            white_bg_folder = images_folder.parent / "images_white_bg"
        white_bg_folder.mkdir(parents=True, exist_ok=True)

    # Find all images
    image_files = sorted(list(images_folder.glob("*.png")))

    if not image_files:
        print(f"No PNG files found in {images_folder}")
        return 0

    print(f"Processing {len(image_files)} images...")

    processed = 0
    skipped = 0

    for img_path in image_files:
        try:
            img = Image.open(img_path)

            # Check if image has alpha channel
            if img.mode != 'RGBA':
                print(f"  Skipping {img_path.name}: No alpha channel (mode: {img.mode})")
                skipped += 1
                continue

            # Extract alpha channel as mask
            r, g, b, a = img.split()

            # Save mask (white = object, black = background)
            mask_path = output_masks_folder / img_path.name
            a.save(mask_path)

            # Create white background version if requested
            if create_white_bg:
                # Create white background
                white_bg = Image.new('RGB', img.size, (255, 255, 255))

                # Composite the image onto white background
                white_bg.paste(img, mask=a)

                white_bg_path = white_bg_folder / img_path.name
                white_bg.save(white_bg_path)

            processed += 1

        except Exception as e:
            print(f"  Error processing {img_path.name}: {e}")
            skipped += 1

    print(f"\nProcessed: {processed}")
    print(f"Skipped: {skipped}")
    print(f"Masks saved to: {output_masks_folder}")

    if create_white_bg:
        print(f"White background images saved to: {white_bg_folder}")

    return processed


def main():
    parser = argparse.ArgumentParser(
        description="Extract masks from RGBA images for 3DGS training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract masks from capture folder
  python extract_masks.py ./capture

  # Also create white background versions
  python extract_masks.py ./capture --create-white-bg

  # Specify custom output folders
  python extract_masks.py ./capture/images --masks-folder ./capture/masks
"""
    )

    parser.add_argument("input", help="Input folder (images folder or parent capture folder)")
    parser.add_argument("--masks-folder", "-m", help="Output folder for masks")
    parser.add_argument("--create-white-bg", action="store_true",
                       help="Also create versions with white background")
    parser.add_argument("--white-bg-folder", "-w", help="Output folder for white background images")

    args = parser.parse_args()

    input_path = Path(args.input).resolve()

    if not input_path.exists():
        print(f"Error: Path does not exist: {input_path}")
        sys.exit(1)

    # Determine images folder
    if input_path.name == "images":
        images_folder = input_path
    elif (input_path / "images").exists():
        images_folder = input_path / "images"
    else:
        # Assume it's the images folder itself
        images_folder = input_path

    masks_folder = Path(args.masks_folder) if args.masks_folder else None
    white_bg_folder = Path(args.white_bg_folder) if args.white_bg_folder else None

    count = extract_masks_from_folder(
        images_folder,
        masks_folder,
        args.create_white_bg,
        white_bg_folder
    )

    if count == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
