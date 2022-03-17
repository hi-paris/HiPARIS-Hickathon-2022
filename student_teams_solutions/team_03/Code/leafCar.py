from libs import preprocessing as pp
from libs.utils import imageToTensor, crop_imgs, draw_img_boxes, box_dataset_generator
from carDetector import carDetector
from carClassifier import carClassifier
import glob
import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torch.nn import Softmax
from torch import nn

import sys

# CML parameters
input_folder = sys.argv[1]
output_file = sys.argv[2]

# Parameters
home = "/home/jovyan/activities_data/hi__paris_2022_hackathon/final_challenge/datasets"

#input_folder = f"{home}/datasets_train/train/"
#output_file = f"./my_work/team-03-084-submission-ab/Code/predictions/predictions.csv"

footprint_path = f"{home}/car_models_footprint.csv"
car_path = f"{home}/datasets_train/train_annotation/_annotation.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier_weights = os.path.abspath('./my_work/team-03-084-submission-ab/Code/models/car_classifier_1.pth')
TOP_K = 5

# Loading images
images_dataset = pp.ImageDataset(input_folder, footprint_path, car_path, num_pixels=512, min_max_scaling=True, with_label=False)

# Loading algorithms
car_detector = carDetector(min_prob=4e-1, car_idx=3)
car_classifier = carClassifier()
car_classifier.load_state_dict(torch.load(classifier_weights))

def oneShotPredictor(image, detector, classifier, dataset, topk=1):
    """
        topk: number of labels to use to predict consumption
    """
    # Getting bbox
    bbox = car_detector.predict(image)[0]
    bbox = [int(x) for x in bbox]

    # Getting label
    if len(bbox) > 0:
        car_image = image[:, :, bbox[1]:bbox[3], bbox[0]:bbox[2]]
        car_labels = nn.Softmax(dim=1)(car_classifier.predict(car_image))

        models_indices, models_weight = torch.topk(car_labels, topk).indices[0], torch.topk(car_labels, topk).values[0]
        models_weight = nn.Softmax(dim=0)(models_weight)
        models_indices, models_weight = models_indices.cpu().numpy(), models_weight.cpu().numpy()
        models_co2 = np.array([dataset.id_to_co2[x] for x in models_indices.tolist()])

        co2_estimation = {
            "max_prob_co2":models_co2[0],
            "weighted_mean":(models_co2*models_weight).sum(),
            "mean":models_co2.mean()
        }

        return bbox, co2_estimation
    else:
        co2_estimation = {
            "max_prob_co2": None,
            "weighted_mean": None,
            "mean": None
        }
        bbox = {"x_min":None, "y_min":None, "x_max":None,"y_max":None}

    return bbox, co2_estimation
    
car_detector.to(DEVICE)
car_classifier.to(DEVICE)

predictions_list = []

for i in range(len(images_dataset)):
    image_path = images_dataset.images_paths[i]
    image = images_dataset[i].to(DEVICE)

    print(f"Processing {image_path}")

    prediction = oneShotPredictor(image, car_detector, car_classifier, images_dataset, topk=TOP_K)
    bbox, co2_consumption = prediction
    bbox = dict(zip(["x_min","y_min", "x_max", "y_max"], bbox))
    
    predictions_list.append((image_path, bbox, co2_consumption))
    
image_path_list = [x[0] for x in predictions_list]
bbox_list = [x[1] for x in predictions_list]
co2_consumption_list = [x[2] for x in predictions_list]

output_df = pd.DataFrame(bbox_list, index=image_path_list).join(
    pd.DataFrame(co2_consumption_list, index=image_path_list)
).reset_index() \
  .rename(columns={
    "index":"filepath",
    "max_prob_co2":"e",
}) \
  .assign(im_name = lambda x: x["filepath"].str.split("/").apply(lambda y : y[-1]))

output_df = output_df[["im_name", "x_min", "y_min", "x_max", "y_max", "e"]].reset_index(drop=True)
mask = output_df["e"].isna()
output_df.loc[mask, :] = 0

output_df.to_csv(output_file, header=True, index=False)