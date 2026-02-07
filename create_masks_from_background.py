#!/usr/bin/env python3
"""
Create masks from images with solid background color.

This script detects the background color (from corners of the image)
and creates masks where background = black, object = white.

Useful for images rendered without transparent background.

Usage:
    python create_masks_from_background.py /path/to/capture
    python create_masks_from_background.py /path/to/capture --threshold 30
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


def detect_background_color(img_array: np.ndarray, sample_size: int = 20) -> np.ndarray:
    """Detect background color from image corners."""
    h, w = img_array.shape[:2]

    # Sample from corners
    corners = [
        img_array[:sample_size, :sample_size],           # top-left
        img_array[:sample_size, -sample_size:],          # top-right
        img_array[-sample_size:, :sample_size],          # bottom-left
        img_array[-sample_size:, -sample_size:],         # bottom-right
    ]

    # Get median color from all corners
    all_samples = np.concatenate([c.reshape(-1, img_array.shape[2]) for c in corners])
    bg_color = np.median(all_samples, axis=0).astype(np.uint8)

    return bg_color


def create_mask(img_array: np.ndarray, bg_color: np.ndarray, threshold: int = 25) -> np.ndarray:
    """Create a mask where object = 255 and background = 0."""
    # Use only RGB channels
    if img_array.shape[2] == 4:
        rgb = img_array[:, :, :3]
    else:
        rgb = img_array

    bg_rgb = bg_color[:3]

    # Calculate distance from background color
    diff = np.abs(rgb.astype(np.float32) - bg_rgb.astype(np.float32))
    distance = np.max(diff, axis=2)  # Max difference across channels

    # Create mask: background where distance is small
    mask = (distance > threshold).astype(np.uint8) * 255

    return mask


def process_folder(images_folder: Path, output_folder: Path = None,
                   threshold: int = 25, preview: bool = False):
    """Process all images in a folder."""

    if output_folder is None:
        output_folder = images_folder.parent / "masks"

    output_folder.mkdir(parents=True, exist_ok=True)

    # Find all images
    image_files = sorted(list(images_folder.glob("*.png")) + list(images_folder.glob("*.jpg")))

    if not image_files:
        print(f"No images found in {images_folder}")
        return 0

    print(f"Processing {len(image_files)} images...")
    print(f"Using background detection threshold: {threshold}")

    # Detect background color from first image
    first_img = np.array(Image.open(image_files[0]))
    bg_color = detect_background_color(first_img)
    print(f"Detected background color: RGB({bg_color[0]}, {bg_color[1]}, {bg_color[2]})")

    processed = 0

    for img_path in image_files:
        try:
            img = Image.open(img_path)
            img_array = np.array(img)

            # Create mask
            mask = create_mask(img_array, bg_color, threshold)

            # Save mask
            mask_img = Image.fromarray(mask, mode='L')
            mask_path = output_folder / f"{img_path.stem}.png"
            mask_img.save(mask_path)

            processed += 1

            if preview and processed == 1:
                # Show first mask for verification
                print(f"\nPreview: First mask saved to {mask_path}")
                print(f"Object pixels: {np.sum(mask == 255)}")
                print(f"Background pixels: {np.sum(mask == 0)}")

        except Exception as e:
            print(f"  Error processing {img_path.name}: {e}")

    print(f"\nProcessed: {processed}")
    print(f"Masks saved to: {output_folder}")

    return processed


def main():
    parser = argparse.ArgumentParser(
        description="Create masks from images with solid background color",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create masks with default threshold
  python create_masks_from_background.py ./capture

  # Adjust threshold (higher = more strict background detection)
  python create_masks_from_background.py ./capture --threshold 40

  # Preview first mask
  python create_masks_from_background.py ./capture --preview
"""
    )

    parser.add_argument("input", help="Input folder (images folder or capture folder)")
    parser.add_argument("--output", "-o", help="Output folder for masks")
    parser.add_argument("--threshold", "-t", type=int, default=25,
                       help="Color difference threshold for background detection (default: 25)")
    parser.add_argument("--preview", "-p", action="store_true",
                       help="Show preview info for first mask")

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
        images_folder = input_path

    output_folder = Path(args.output) if args.output else None

    count = process_folder(images_folder, output_folder, args.threshold, args.preview)

    if count == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
