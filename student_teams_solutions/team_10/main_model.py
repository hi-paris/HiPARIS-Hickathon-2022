from car_detector import get_output_layers, extract_car
from dataset_extractor import (
    download_dataset,
    annotations_formatting,
    cropping_organizing_directory,
    recup_hp_data,
)
from filtering import apply_filter, create_filtered_dataset
from classifier import create_data, launch_training, predict_emission
import numpy as np
import pandas as pd
import PIL
import os
import shutil


# INSERT : GLOBAL DIRECTORY
global_dir = "/home/jovyan/data/End-to-end model/"
# INSERT : PATH WITH IMAGES TO PREDICT
dataset_predicted_folder = "/home/jovyan/activities_data/hi__paris_2022_hackathon/final_challenge/datasets/datasets_test/test/"
# INSERT : PATH WITH PROVIDED TRAIN/TEST SETS
original_dataset = "/home/jovyan/activities_data/hi__paris_2022_hackathon/final_challenge/datasets/datasets_train/"

# DATA DIRECTORIES
data_dir = global_dir + "data_dir/"
dataset_dir = data_dir + "dataset/"
reference_dataset = data_dir + "car_models_footprint.csv"
model_dir = global_dir + "model_dir/"
yolo_dir = global_dir + "model_yolo/"


Training = False
Predicting = True


batch_size_train = 128
nb_car_models = 100
lr = 0.01
momentum = 0.9
patience = 3
threshold = 0.9
nb_epochs = 10
print_every = 40
topk = 10
model_name = "model.pth"


"""
COMPLETING THE DATASET
"""
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
if not os.path.exists(data_dir):
    os.mkdir(data_dir)
    src = "/home/jovyan/activities_data/hi__paris_2022_hackathon/final_challenge/datasets/car_models_footprint.csv"
    shutil.copyfile(src, reference_dataset)
if not os.path.exists(dataset_predicted_folder):
    os.mkdir(dataset_predicted_folder)

if not os.path.exists(dataset_dir):
    os.mkdir(dataset_dir)
    ref = reference_dataset
    download_dataset(dataset_dir)
    annotations_formatting(dataset_dir)
    cropping_organizing_directory(dataset_dir, ref)
    recup_hp_data(dataset_dir, original_dataset)


dataset_train_folder = dataset_dir + "train_set/"
dataset_test_folder = dataset_dir + "test_set/"

dct = create_data(dataset_train_folder, dataset_test_folder, data_dir)

"""
TRAINING CLASSIFIER
"""

if Training:

    launch_training(
        trainloader=dct["trainloader"],
        validloader=dct["validloader"],
        lr=lr,
        nb_epochs=nb_epochs,
        momentum=momentum,
        patience=patience,
        threshold=threshold,
        print_every=print_every,
        class_to_idx=dct["class_to_idx"],
        model_save_dir=model_dir + model_name,
        nb_car_models=nb_car_models,
    )


"""
CAR DETECTOR AND PREDICTION
"""

if Predicting:
    predictions = []
    filenames_to_predict = os.listdir(dataset_predicted_folder)
    for img_filenames in filenames_to_predict:
        if img_filenames[-4:] == ".jpg":
            list_box, list_img = extract_car(
                dataset_predicted_folder + img_filenames, yolo_dir, verbose=False
            )
            if len(list_box) >= 1:
                xmins, xmaxs = [0] * len(list_box), [0] * len(list_box)
                ymins, ymaxs = [0] * len(list_box), [0] * len(list_box)
                for i in range(len(list_box)):
                    xmins[i], xmaxs[i], ymins[i], ymaxs[i] = (
                        list_box[i][0],
                        list_box[i][0] + list_box[i][2],
                        list_box[i][1],
                        list_box[i][1] + list_box[i][3],
                    )
                areas = [
                    (ymaxs[i] - ymins[i]) * (xmaxs[i] - xmins[i])
                    for i in range(len(list_box))
                ]
                s = np.argmax(areas)
                xmin, xmax, ymin, ymax = xmins[s], xmaxs[s], ymins[s], ymaxs[s]
                image = list_img[s]
                try:
                    emission = predict_emission(
                        image,
                        model_dir + model_name,
                        topk=topk,
                        models_emissions=dct["models_emissions"],
                        nb_car_models=nb_car_models,
                        classes=dct["classes"],
                    )
                    predictions.append(
                        [
                            img_filenames,
                            int(xmin),
                            int(ymin),
                            int(xmax),
                            int(ymax),
                            "car",
                            emission,
                        ]
                    )
                except:
                    predictions.append([img_filenames, 0, 0, 0, 0, "no car", 0])
            else:
                predictions.append([img_filenames, 0, 0, 0, 0, "no car", 0])

        df = pd.DataFrame(
            predictions,
            columns=[
                "im_name",
                "x_min",
                "y_min",
                "x_max",
                "y_max",
                "class",
                "emission",
            ],
        )
        df.to_csv(global_dir + "predictions.csv", sep=";", index=False)
