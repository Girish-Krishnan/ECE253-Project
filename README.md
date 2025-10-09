# ECE 253 Project

Girish Krishnan, Philip Pincencia, and Yusuf Morsi

## Public Code

The following directories contain public code that we will use in this project:

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