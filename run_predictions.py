import os
import numpy as np
import json
from PIL import Image
from tqdm import tqdm

# This function imports the kernel we will use for matched filtering
def import_kernel():
    # read image using PIL:
    I = Image.open("kernel.jpg")

    # convert to numpy array:
    I = np.asarray(I)
    return I


def detect_red_light(I):
    '''
    This function takes a numpy array <I> and returns a list <bounding_boxes>.
    The list <bounding_boxes> should have one element for each red light in the
    image. Each element of <bounding_boxes> should itself be a list, containing
    four integers that specify a bounding box: the row and column index of the
    top left corner and the row and column index of the bottom right corner (in
    that order). See the code below for an example.

    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''


    bounding_boxes = [] # This should be a list of lists, each of length 4. See format example below.

    # Import the kernel
    kernel = import_kernel()
    box_height, box_width, n_channels = kernel.shape

    # Convert kernel channels into normalized vectors
    img_flatten = I.flatten()
    img_mean = np.mean(img_flatten)
    img_std = np.std(img_flatten)

    kernel_ch1 = kernel[:,:,0].flatten()
    kernel_ch1 = (kernel_ch1 - img_mean) / img_std
    kernel_ch1 = kernel_ch1 / np.linalg.norm(kernel_ch1)

    kernel_ch2 = kernel[:,:,1].flatten()
    kernel_ch2 = (kernel_ch2 - img_mean) / img_std
    kernel_ch2 = kernel_ch2 / np.linalg.norm(kernel_ch2)

    kernel_ch3 = kernel[:,:,2].flatten()
    kernel_ch3 = (kernel_ch3 - img_mean) / img_std
    kernel_ch3 = kernel_ch3 / np.linalg.norm(kernel_ch3)

    kernels = [kernel_ch1, kernel_ch2, kernel_ch3]

    # Threshold for matched filtering
    T = 0.7

    # upper_left will keep track of where we are in our scan
    upper_left = [0,0]
    while(True):
        # These if statements check to ensure our "sliding window" is in bounds
        if upper_left[1] >= I.shape[1]-box_width:
            upper_left[1] = 0
            upper_left[0] += 1

        if upper_left[0] >= I.shape[0]-box_height:
            break

        # Set is_traffic_light to false once we detect a mismatch
        is_traffic_light = True

        # Run matched filtering for each channel
        for n in range(n_channels):
            # Current "sliding window" section of the photo
            right_col = upper_left[0]+box_height
            bottom_row = upper_left[1]+box_width
            curr = I[upper_left[0]:right_col,
            upper_left[1]:bottom_row,n]

            # convert curr to normalized vector
            curr = curr.flatten()
            curr = (curr - img_mean) / img_std
            curr = curr / np.linalg.norm(curr)

            # Compute inner product
            ip = np.inner(curr, kernels[n])

            if ip < T:
                is_traffic_light = False
                break

        if is_traffic_light:
            # This section of the image passed the threshold for every channel
            tl_row = upper_left[0]
            tl_col = upper_left[1]
            br_row = upper_left[0]+box_height
            br_col = upper_left[1]+box_width
            bounding_boxes.append([tl_row,tl_col,br_row,br_col])
            # Move our sliding window box_width pixels to the right
            # Since we don't want to detect the same stoplight again
            upper_left = [upper_left[0], upper_left[1]+box_width]
        else:
            # Move our sliding window one pixel to the right
            upper_left = [upper_left[0], upper_left[1]+1]

    for i in range(len(bounding_boxes)):
        assert len(bounding_boxes[i]) == 4

    return bounding_boxes

# set the path to the downloaded data:
data_path = './RedLights2011_Medium'

# set a path for saving predictions:
preds_path = './hw01_preds'
os.makedirs(preds_path,exist_ok=True) # create directory if needed

# get sorted list of files:
file_names = sorted(os.listdir(data_path))

# remove any non-JPEG files and visualizations
file_names = [f for f in file_names if ('.jpg' in f) and (not "result" in f)]

preds = {}
for i in tqdm(range(len(file_names))):

    # read image using PIL:
    I = Image.open(os.path.join(data_path,file_names[i]))

    # convert to numpy array:
    I = np.asarray(I)

    preds[file_names[i]] = detect_red_light(I)

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds.json'),'w') as f:
    json.dump(preds,f)
