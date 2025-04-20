import numpy as np
import pandas as pd
import os, re, cv2
from sklearn.model_selection import train_test_split as train_test_split_

# Load and resize images
def extract_number(name):
    match = re.search(r'\d+', name)
    return int(match.group()) if match else float('inf')

def sort_key(name):
    return [int(part) for part in re.findall(r'\d+', name)]

def resize_image(image,target_size):
    # Get current image dimensions
    h, w = image.shape[:2]

    # Determine if we need to upscale or downscale
    if w < target_size[0] or h < target_size[1]:
        interpolation = cv2.INTER_CUBIC  # Good for upscaling
    else:
        interpolation = cv2.INTER_AREA  # Good for downscaling

    resized_image = cv2.resize(image, target_size, interpolation=interpolation)
    return resized_image


def get_images(base_path='./ImagesFolder/',target_size = (224, 224),bgr_flag =1,preprocess=None):
    img_pathlist = os.listdir(base_path)
    img_pathlist = [os.path.join(base_path, f) for f in img_pathlist]
    sorted_paths = sorted(img_pathlist, key=lambda x: extract_number(os.path.basename(x)))
    images = []
    for path in sorted_paths:
        img = cv2.imread(path,bgr_flag)

        # get preprocess function
        if preprocess:
            img = preprocess(img)

        # resize
        img = resize_image(img, target_size)
        images.append(img)
    return np.array(images)

def get_target(target_path='label.csv', use_header=False):
    header = 0 if use_header else None
    target = pd.read_csv(target_path, header=header).values.flatten()
    return target

def train_test_split(X, y,random_state=42,train_size=0.9):
    X_train, X_test, y_train, y_test = train_test_split_(X, y, random_state=random_state, train_size=train_size)
    return X_train, X_test, y_train, y_test

def get_data(images_path='./ImagesFolder/',
             target_size = (224, 224),
             bgr_flag =1,
             preprocess=None,
             target_path='label.csv',
             use_header=False,
             split=True,
             random_state=42,
             train_size=0.9):
    images,target = get_images(images_path,target_size,bgr_flag,preprocess), get_target(target_path, use_header)

    if split:
        # return X_train, X_test, y_train, y_test
        return train_test_split(images,target,random_state,train_size)

    return images,target


