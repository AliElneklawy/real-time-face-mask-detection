# This script will be used to check that all the images are healthy and not broken
# using two methods; Pillow and open-cv

import cv2
import os
from PIL import Image


def is_valid_image(file_path):
    validations = []

    # Method 1: using Pillow
    try:
        with Image.open(file_path) as img:
            img.verify()
        validations.append(True)
    except Exception as e:
        validations.append(False)

    # Method 2: using open-cv
    try:
        img = cv2.imread(file_path)
        if img is not None:
            validations.append(True)
        else:
            validations.append(False)
    except Exception as e:
        validations.append(False)

    return all(validations)
    
def check_image_directory(directory_path, verbose=1):
    not_valid = []
    for i, filename in enumerate(os.listdir(directory_path)):
        if verbose: print(f'Checking image: {i+1}')
        if filename.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')):
            file_path = os.path.join(directory_path, filename)
            if not is_valid_image(file_path):
                not_valid.append(file_path)
                print(f"Broken image: {file_path}")

    if not not_valid:
        print('All OK')
    else:
        print(f'These images are broken: {not_valid}')

if __name__ == "__main__":
    dirs = [os.path.join('aug', 'with_mask'), os.path.join('aug', 'without_mask')]
    for dir in dirs:
        check_image_directory(dir, verbose=1)


