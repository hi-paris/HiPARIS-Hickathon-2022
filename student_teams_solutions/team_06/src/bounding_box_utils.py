import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os


class BoundingBox():
    """BoundingBox output from Yolo inference"""

    def __init__(self, path: str, serie: pd.Series):
        """
        Parameters
        -----------
        path: String, path to the image inference
        serie: pd.Series, Yolo output in format 
            c=class, 
            bx=boundin box center x
            by=bounding box _center y
            bh=bounding box height
            bw=bounding box width
            conf=confidence
        """
        self.image = Image.open(path)
        self.c = serie.c
        self.bx = serie.bx
        self.by = serie.by
        self.bh = serie.bh
        self.bw = serie.bw
        self.conf = serie.conf

    def bb_plot(self):
        """Plot detected bounding box"""
        imw = self.image.size[0]
        imh = self.image.size[1]

        bottom_x = (self.bx - (self.bw / 2))*imw
        bottom_y = (self.by - (self.bh / 2))*imh
        _, axe = plt.subplots()
        axe.imshow(self.image)
        # Create a Rectangle patch
        rect = patches.Rectangle(
            (bottom_x, bottom_y),
            self.bw * imw,
            self.bh * imh,
            linewidth=1,
            edgecolor="r",
            facecolor="none"
        )
        # Add the patch to the Axes
        axe.add_patch(rect)
        plt.show()

    def to_evaluation(self):
        """Transform yolo's bounding box format to the evaluation format"""
        imw = self.image.size[0]
        imh = self.image.size[1]
        # get locations xmin, ymin, xmax, ymax
        xmin = (self.bx - (self.bw / 2))*imw
        ymin = (self.by - (self.bh / 2))*imh
        xmax = (self.bx + (self.bw / 2))*imw
        ymax = (self.by + (self.bh / 2))*imh
        return pd.Series({
            "x_min": xmin,
            "y_min": ymin,
            "x_max": xmax,
            "y_max": ymax})


def biggest_box(pred_df):
    """Returns: pd.Series, inferences of the highest area box"""
    return pred_df.sort_values("area", ascending=False).iloc[0,:]


def select_inference_from_file(img_name, images_path, detect_dir):
    """Select the car's inference with the maximal area
    Parameter : file name with .jpg extension
    Returns: Class BoundingBox"""

    def area(box):
        """Compute the area
        Parameter: BoundingBox
        Return: Float """
        return box.bw * box.bh

    # delete the image extention (.jpg) and replace with .txt
    pred_file = os.path.splitext(img_name)[0]+".txt"
    pred_path = os.path.join(detect_dir, pred_file)
    cols = ["c", "bx", "by", "bw", "bh", "conf"]
    try:
        pred_df = pd.read_csv(
            pred_path,
            sep=" ",
            names=cols
        )
        pred_df = pred_df[(pred_df.c == 2) | (pred_df.c == 7)]
        # Test if inference predicted the class car or truck (index=2 and 7)
        if len(pred_df) > 0:
            # If there is more than one inference, select the box with the biggest area
            if len(pred_df) > 1:
                pred_df["area"] = pred_df.apply(area, axis=1)
                inference = biggest_box(pred_df)
            # If not, select the first instance
            else:
                inference = pred_df.iloc[0,:]
        # if there is no car predicted, return a null vector
        else:
            inference = pd.Series(
                np.zeros(6),
                index=cols
            )
    except FileNotFoundError:
        # print("Yolo didn't make any inference on this image")
        inference = pd.Series(
            np.zeros(6),
            index=cols
        )
    return BoundingBox(os.path.join(images_path, img_name), inference)


def predict_pipeline(images_path, detect_dir):
    """Final pipeline
    Returns a dataframe with bounding box position and energy consumption"""
    images = os.listdir(images_path)
    cols = ["im_name", "x_min", "y_min", "x_max", "y_max", " e"]
    results_df = pd.DataFrame(
        np.zeros((len(images), len(cols))), columns=cols)
    results_df["im_name"] = images
    results_df = results_df.set_index("im_name")
    for img_name in images:
        if os.path.splitext(img_name)[-1] == ".jpg":
            predicted_box = select_inference_from_file(img_name, images_path, detect_dir)
            predicted_box = predicted_box.to_evaluation()
            results_df.at[img_name, "x_min"] = predicted_box.x_min
            results_df.at[img_name, "x_max"] = predicted_box.x_max
            results_df.at[img_name, "y_min"] = predicted_box.y_min
            results_df.at[img_name, "y_max"] = predicted_box.y_max
    results_df = results_df.reset_index()
    return results_df

# IMAGES_PATH = "../../../datasets/datasets_test/test/"
# DETECT_DIR = "./runs/detect/test/labels/"
# results_csv = predict_pipeline(IMAGE_PATH, DETECT_DIR)
# results_csv.to_csv("results.csv", index=False)
