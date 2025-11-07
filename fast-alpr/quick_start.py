import cv2

from fast_alpr import ALPR

# Initialize the ALPR
alpr = ALPR(
    detector_model="yolo-v9-t-384-license-plate-end2end",
    ocr_model="cct-xs-v1-global-model",
)

# Load the image
image_path = "../data/testing/image_defoggy.png"
frame = cv2.imread(image_path)

# Draw predictions on the image
annotated_frame = alpr.draw_predictions(frame)
cv2.imshow("Annotated Frame", annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()