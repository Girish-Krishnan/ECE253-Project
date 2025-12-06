# Robust Image Restoration for Adverse-Condition License Plate Recognition

Girish Krishnan, Philip Pincencia, and Yusuf Morsi


## Dataset

The dataset that we manually collected and used in this project is available on Kaggle:

https://www.kaggle.com/datasets/yusufmorsi/adverse-condition-image-restoration-for-lpr


## Public Code

The following directories contain public code that we will use in this project:

### License Plate Detection
* `fast-alpr/`: This contains the code for the overall system that integrates license plate detection (via YOLO bounding boxes) and license plate recognition (via OCR).
* `open-image-models/`: This contains the code for the model we will use to detect license plates in images. The output is a bounding box around the license plate.
* `fast-plate-ocr/`: This contains the code for the OCR model that will take in a cropped image of a license plate and output the text on the license plate.

### Image Processing
* `algorithms/motion-deblur/`: This contains the motion deblur algorithm implementation (`motion_deblur_py.py`). This implements classical blind motion deblurring using Richardson-Lucy deconvolution with grid search over motion kernels, plus a Wiener filter as a baseline. The algorithm performs grid search over PSF parameter space (length and angle) to find the best motion blur kernel, then applies either Richardson-Lucy deconvolution or Wiener filter deconvolution to restore the image.
* `algorithms/iagcwd/`: This contains the Improved Adaptive Gamma Correction with Weighted Distribution (IAGCWD) algorithm implementation (`IAGCWD.py`). This algorithm improves the contrast of brightness-distorted images adaptively using improved adaptive gamma correction. It is used for enhancing underexposed or low-light images.
* `algorithms/dcp-dehaze/`: This contains the Dark Channel Prior (DCP) based image dehazing algorithm implementation. This implements single image haze removal using the dark channel prior method proposed by He et al. The algorithm estimates atmospheric light and transmission map to recover haze-free images from hazy inputs.

Note that the code in `fast-alpr/` essentially integrates the other two repositories together. But, I've still included the other two repositories because they contain code to help fine-tune the models on our custom dataset, as well as the results of pre-training the models on public datasets.

On all three repositories, you can install dependencies by `cd`ing into each directory separately and running:

```bash
make install
make checks
```

Then, you will notice that you get a `.venv` directory in each of the three directories. You can activate the virtual environment (in each directory separately) by running:

```bash
source .venv/bin/activate
```

## Utility Scripts

### `scripts/frame_scrubber.py`

This tool lets you step through video frames, view them interactively, and save specific frames as images.
    
#### Basic Usage

```bash
python scripts/frame_scrubber.py -v path/to/video.mov -o ./images
```

#### Controls

* **Slider:** Drag to jump to any frame in the video.
* **Frame Box:** Enter a frame number and press **Enter** or click **Go**.
* **Prev / Next:** Move one frame backward or forward.
* **Save JPG:** Save the current frame as a JPG image in the specified output directory. Files are saved as `videoName_frame_000123.jpg`.
* **Rotate 90°:** Rotate the video view and saved images clockwise by 90 degrees.
* **Left / Right Arrow Keys:** Step through frames.
* **Spacebar:** Save the current frame.

## LaTeX Code (for project proposal and report)

See the directory `project-proposal/` for the LaTeX code used to generate the project proposal.