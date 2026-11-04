"""
Compute the grayscale pixel fraction for an image, or for every image in a folder
(including subfolders),
using MegaDetector's gray_scale_fraction (a rough night-time/IR image indicator).

Usage:
    python run_gray_scale_fraction.py <image_or_folder> <output.csv> [--crop-top 0.1] [--crop-bottom 0.1]
"""

import argparse
import csv
import os

from megadetector.visualization.visualization_utils import gray_scale_fraction

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='Image file or folder of images')
    parser.add_argument('output', help='Path of the CSV file to write')
    parser.add_argument('--crop-top', type=float, default=0.1,
                        help='Fraction of image height to ignore at the top (default 0.1)')
    parser.add_argument('--crop-bottom', type=float, default=0.1,
                        help='Fraction of image height to ignore at the bottom (default 0.1)')
    args = parser.parse_args()

    if os.path.isdir(args.path):
        image_files = sorted(
            os.path.join(root, f)
            for root, _, files in os.walk(args.path)
            for f in files
            if f.lower().endswith(IMAGE_EXTENSIONS))
    else:
        image_files = [args.path]

    with open(args.output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'gray_scale_fraction', 'error'])

        for image_file in image_files:
            # Record unreadable images instead of crashing, so one bad file
            # doesn't kill a long job
            try:
                fraction = gray_scale_fraction(image_file, crop_size=(args.crop_top, args.crop_bottom))
                writer.writerow([image_file, '{:.4f}'.format(fraction), ''])
            except Exception as e:
                writer.writerow([image_file, '', str(e)])

    print('Wrote {} rows to {}'.format(len(image_files), args.output))


if __name__ == '__main__':
    main()
