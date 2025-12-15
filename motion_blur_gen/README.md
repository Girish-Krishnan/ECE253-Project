# Directory Structure
The script expects an input image directory and writes results to an output directory. Both paths are defined relative to the project root.

# Configuration
All parameters are defined in `blur_config.yaml`. This includes input and output directories and detailed blur behavior such as blur strength, angle range, kernel intensity, color channel shifts, noise level, and how much of the original image is blended back in. The configuration is validated using pydantic to catch invalid values early.

# How It Works
For each image, a random blur degree and angle are sampled from the configured ranges. A motion kernel is generated, optionally following a curved trajectory to mimic camera shake. The kernel is rotated, applied to the image, optionally varied per color channel, followed by noise injection and blending with the original image.

# Usage
Place your images in the configured input directory, then run the script from the project root.

```
python main.py
```

The script will read `blur_config.yaml`, process all supported images, and save blurred versions to the output directory with a `_blurred` suffix.