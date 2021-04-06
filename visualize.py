import json
from PIL import Image
import os
import numpy as np

preds_path = "hw01_preds/preds.json"
data_path = './RedLights2011_Medium'

def main():
    with open(preds_path) as f:
        preds = json.load(f)

    for filename, boxes in preds.items():
        # Get image
        I = Image.open(os.path.join(data_path,filename))

        # convert to numpy array:
        I = np.asarray(I).copy()

        for box in boxes:
            tl_row, tl_col, br_row, br_col = box

            # Left and right sides of box
            for i in range(tl_row, br_row+1):
                I[i][tl_col] = [255,255,255]
                I[i][br_col] = [255,255,255]

            # Top and bottom sides of box
            for j in range(tl_col, br_col+1):
                I[tl_row][j] = [255,255,255]
                I[br_row][j] = [255,255,255]

        # Save image with boxes
        img = Image.fromarray(I, 'RGB')
        img.save(os.path.join(data_path, filename.split(".")[0]) + "_result.jpg")


if __name__ == "__main__":
    main()
