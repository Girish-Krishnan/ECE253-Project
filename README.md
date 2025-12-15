# Robust Image Restoration for Adverse-Condition License Plate Recognition

Girish Krishnan, Philip Pincencia, and Yusuf Morsi


## Dataset

The dataset that we manually collected and used in this project is available on Kaggle:

https://www.kaggle.com/datasets/yusufmorsi/adverse-condition-image-restoration-for-lpr

This contains 200 images of each category: motion-blurred, low-light, and foggy images. (600 in total)

## Public Code

The following directories contain public code that we will use in this project:

### License Plate Detection

Note that the code in these three directories are from separate public repositories, and each contains its own `README.md` file with more details about the respective algorithms and how to run them.

* `fast-alpr/`: This contains the code for the overall system that integrates license plate detection (via YOLO bounding boxes) and license plate recognition (via OCR).
* `open-image-models/`: This contains the code for the model we will use to detect license plates in images. The output is a bounding box around the license plate.
* `fast-plate-ocr/`: This contains the code for the OCR model that will take in a cropped image of a license plate and output the text on the license plate.

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

### Image Processing

Note that each of these directories contains a `README.md` file with more details about the respective algorithms and how to run them.

* `algorithms/motion-deblur/`: This contains the motion deblur algorithm implementation (`motion_deblur_py.py`). This implements classical blind motion deblurring using Richardson-Lucy deconvolution with grid search over motion kernels, plus a Wiener filter as a baseline. The algorithm performs grid search over PSF parameter space (length and angle) to find the best motion blur kernel, then applies either Richardson-Lucy deconvolution or Wiener filter deconvolution to restore the image.
* `algorithms/iagcwd/`: This contains the Improved Adaptive Gamma Correction with Weighted Distribution (IAGCWD) algorithm implementation (`IAGCWD.py`). This algorithm improves the contrast of brightness-distorted images adaptively using improved adaptive gamma correction. It is used for enhancing underexposed or low-light images.
* `algorithms/Low-light-Image-Enhancement/`: This contains the Retinex-based low-light image enhancement algorithm implementation (from the paper Dual Illumination Estimation for Robust Exposure Correction). This implements a method for enhancing low-light images based on the Retinex theory, which decomposes an image into illumination and reflectance components to improve visibility in dark regions.
* `algorithms/dcp-dehaze/`: This contains the Dark Channel Prior (DCP) based image dehazing algorithm implementation. This implements single image haze removal using the dark channel prior method proposed by He et al. The algorithm estimates atmospheric light and transmission map to recover haze-free images from hazy inputs.
* `algorithms/DehazeNet_Pytorch/`: This contains the DehazeNet deep learning based image dehazing algorithm implementation. DehazeNet is a convolutional neural network designed for single image haze removal. The model learns to estimate the transmission map from hazy images, which is then used to recover the haze-free image.

## Utility Scripts

### `scripts/frame_scrubber.py`

This tool lets you step through video frames, view them interactively, and save specific frames as images.

This is useful when you take a video and want to break it down into individual frames for labeling or processing.

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

### `scripts/data_labeler.py`

This tool allows you to open up images present in a given directory, and label them with text labels indicating the license plate text.

```bash
python scripts/data_labeler.py ./path/to/images --csv ./path/to/labels.txt
```

If the labels file already exists, it will load the existing labels and allow you to edit them. If it does not exist, it will create a new one.

### `scripts/distortion_classifier.py`

This script classifies images in a given directory into three categories: motion-blurred, low-light, and foggy images. It uses simple image processing techniques to analyze the images and categorize them based on their characteristics.

```bash
python scripts/distortion_classifier.py --img ./path/to/images
```

### `scripts/fast_alpr_wrapper.py`

This script wraps the `fast-alpr` repository into a Python script to make it easier to use. It takes in an image file and outputs the results of the `fast-alpr` repository.

```bash
python scripts/fast_alpr_wrapper.py
```

### `scripts/evaluation_metrics.py`

This script implements evaluation metrics such as NIQE, BRISQUE, Sharpness, Brightness, and also uses the fast ALPR wrapper to compute accuracy and confidence scores.

```bash
python scripts/evaluation_metrics.py --raw_dir ./path/to/raw_images --processed_dir ./path/to/restored_images --labels_csv ./path/to/labels.csv --results_csv ./path/to/results.csv --brisque_model ./misc/brisque_model_live.yml --brisque_range ./misc/brisque_range_live.yml
```

## Jupyter Notebook

The Jupyter notebook `license_plate_restoration.ipynb` contains the code that allows you to pass in a directory containing certain images, and this runs our classifier to decide which images need which restoration algorithms, applies the appropriate restoration algorithms, and then evaluates the results.

## LaTeX Code (for project proposal and report)

See the directory `latex/` for the LaTeX code used to generate the project proposal and report.