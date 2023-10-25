import albumentations as A
import cv2
from PIL import Image
import numpy as np
import os

if not os.path.exists('aug'): #directory to save augmented images
    os.makedirs('aug')

if not os.path.exists('aug//with_mask'): #directory to save masked images
    os.makedirs('aug//with_mask')

if not os.path.exists('aug//without_mask'): #directory to save unmasked images
    os.makedirs('aug//without_mask')
print("Created directories....")


IMG_WIDTH, IMG_HEIGHT, AUG_IMG_NUM = 224, 224, 2 #AUG_IMG_NUM: generate AUG_IMG_NUM examples for each image
images = []
input_directory_with, input_directory_without = os.path.join('im', 'with_mask'), os.path.join('im', 'without_mask')
output_directory_with, output_directory_without = os.path.join('aug', 'with_mask'), os.path.join('aug', 'without_mask')
dirs_arr = [(input_directory_with, output_directory_with), 
            (input_directory_without, output_directory_without)]
aug_im_counter = 0

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
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.9),
        A.RandomRain(p=0.5),
    ]
)
print("Defined transformations....")

def augment(input_dir, output_dir):
    print('Performing augmentation...')
    aug_im_counter = 1

    for im in os.listdir(input_dir):
        image = Image.open(os.path.join(input_dir, im))
        image = np.array(image)

        for _ in range(AUG_IMG_NUM):
            output_im = os.path.join(output_dir, 'augmented_image_' + str(aug_im_counter) + '.png')

            if not os.path.exists(output_im):
                
                try:  #skip the images that will cause any problem and go to the next one.
                    augmentations = transform(image=image)
                except:
                    print('Error!')
                    break

                aug_im = augmentations['image']
                augmented_image = Image.fromarray(aug_im)
                augmented_image.save(output_im)

            print(f'Image {aug_im_counter} done...')
            aug_im_counter += 1    

for with_dir, without_dir in dirs_arr:
    augment(with_dir, without_dir)
