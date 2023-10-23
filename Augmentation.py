import albumentations as A
import cv2
from PIL import Image
import numpy as np
import os

IMG_WIDTH, IMG_HEIGHT, AUG_IMG_NUM = 250, 250, 20
images = []
input_directory, output_directory = 'im', 'aug'
im_counter = 0

transform = A.Compose(
    [
        A.Resize(width=IMG_WIDTH, height=IMG_HEIGHT),
        A.Rotate(p=0.3, border_mode=cv2.BORDER_CONSTANT),
        A.HorizontalFlip(p=0.9),
        A.VerticalFlip(p=0.2),
        A.RGBShift(25, 25, 25, p=0.9),
        A.Blur(blur_limit=3, p=0.6),
        A.Compose([
            A.Blur(blur_limit=3, p=0.6),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.9),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.9),
            ], p=0.9),
        ], p=0.9),
        A.RandomShadow(p=0.5),
        A.RandomFog(p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.9),
        A.RandomRain(p=0.5),
    ]
)

if not os.path.exists(output_directory): #directory to save augmented images
    os.makedirs(output_directory)


for im in os.listdir(input_directory):
    image = Image.open(os.path.join(input_directory, im))
    image = np.array(image)

    for _ in range(AUG_IMG_NUM):
        augmentations = transform(image=image)
        aug_im = augmentations['image']

        output_path = os.path.join(output_directory, 'augmented_image_' + str(im_counter) + '.png')
        augmented_image = Image.fromarray(aug_im)
        augmented_image.save(output_path)

        im_counter += 1
