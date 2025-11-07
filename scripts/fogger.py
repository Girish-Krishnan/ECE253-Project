import imgaug.augmenters as iaa
import imageio.v3 as iio

image = iio.imread("fog5.png")  # Replace with your image path
fog_augmenter = iaa.Fog()
foggy_image = fog_augmenter(image=image)
iio.imwrite("image_foggy.png", foggy_image)