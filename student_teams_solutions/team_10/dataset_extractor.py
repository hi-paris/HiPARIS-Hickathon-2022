import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
from collections import Counter
import os
from urllib.request import urlretrieve
import tarfile
from scipy.io import loadmat
import shutil


def download_dataset(path):
    """
    Allows to download the dataset from internet in the path folder, and decompress it
    """
    if not os.path.exists(path):
        os.makedirs(path)
    # Downloading images
    url = "http://ai.stanford.edu/~jkrause/car196/car_ims.tgz"
    file = "car_ims.tgz"
    target = path + file
    urlretrieve(url, target)
    tarfile.open(target, "r:gz").extractall(path=path)

    # Downloading annotations
    url = "http://ai.stanford.edu/~jkrause/car196/cars_annos.mat"
    file = "annotations.mat"
    target = path + file
    urlretrieve(url, target)


def annotations_formatting(path):
    """
    Formats the annotations downloaded and stored in the path, and returns 3 csv:
        - one contains the annotations for all images,
        - one contains the annotations for the train set,
        - the last one contains the annotations for the test set
    """
    file = "annotations.mat"
    target = path + file
    annotations = loadmat(target)
    l_models = [x[0] for x in annotations["class_names"][0]]

    t = np.vstack(
        [
            [
                ligne[i][0].split("/")[-1] if i == 0 else ligne[i][0][0]
                for i in range(len(ligne))
            ]
            for ligne in annotations["annotations"][0]
        ]
    )

    df_annotations = pd.DataFrame(
        t, columns=["fname", "x_min", "y_min", "x_max", "y_max", "models", "test"]
    )
    df_annotations["models"] = df_annotations.apply(
        lambda x: l_models[int(x["models"]) - 1], axis=1
    )

    df_annotations = df_annotations.sort_values(["models"], ignore_index=True)

    target = path + "annotations_global.csv"
    df_annotations.to_csv(target, sep=";", index=False)

    df_train = (
        df_annotations[df_annotations["test"] == 0]
        .copy()
        .reset_index(drop=True)
        .drop(columns=["test"])
    )
    target = path + "annotations_train.csv"
    df_train.to_csv(target, sep=";", index=False)

    df_test = (
        df_annotations[df_annotations["test"] == 0]
        .copy()
        .reset_index(drop=True)
        .drop(columns=["test"])
    )
    target = path + "annotations_test.csv"
    df_test.to_csv(target, sep=";", index=False)


def cropping_organizing_directory(path, ref):
    target = path + "annotations_global.csv"
    df_global = pd.read_csv(target, sep=";")

    path_train = path + "train_set/"

    path_test = path + "test_set/"

    if not os.path.exists(path_train):
        os.makedirs(path_train)
    if not os.path.exists(path_test):
        os.makedirs(path_test)

    l_models = pd.read_csv(ref, sep=";")["models"].unique()

    for i in range(len(df_global)):
        model = df_global.at[i, "models"]
        if model in l_models:
            file = df_global.at[i, "fname"]
            test = df_global.at[i, "test"]
            x_min = df_global.at[i, "x_min"]
            y_min = df_global.at[i, "y_min"]

            x_max = df_global.at[i, "x_max"]
            y_max = df_global.at[i, "y_max"]

            if test:
                target = path_test + model
            else:
                target = path_train + model
            if not os.path.exists(target):
                os.makedirs(target)

            target = target + "/" + file

            img = path + "car_ims/" + file
            img = np.asanyarray(Image.open(img))
            if img.shape == 3:
                img_crop = img[y_min:y_max, x_min:x_max, :]
            else:
                img_crop = img[y_min:y_max, x_min:x_max]
            img_crop = Image.fromarray(img_crop)
            img_crop.save(target)

    # Removal of original images
    shutil.rmtree(path + "car_ims/")
    os.remove(path + "annotations.mat")
    os.remove(path + "car_ims.tgz")


def recup_hp_data(path, src):
    # récup images train
    annotations = src + "train_annotation/_annotation.csv"
    df_annotations = pd.read_csv(annotations, sep=",", index_col=0)
    df_annotations = df_annotations[df_annotations["class"] == "car"].sort_values(
        ["models", "im_name"], ignore_index=True
    )
    for i in range(len(df_annotations)):
        model = df_annotations.at[i, "models"]
        file = df_annotations.at[i, "im_name"]
        img = src + "train/" + file
        x_min = df_annotations.at[i, "x_min"]
        y_min = df_annotations.at[i, "y_min"]
        x_max = df_annotations.at[i, "x_max"]
        y_max = df_annotations.at[i, "y_max"]
        if x_min + y_min + x_max + y_max > 0:
            x_min = int(x_min)
            y_min = int(y_min)
            x_max = int(x_max)
            y_max = int(y_max)
            tgt = path + "train_set/" + model + "/" + file
            img = np.asanyarray(Image.open(img))
            if img.shape == 3:
                img_crop = img[y_min:y_max, x_min:x_max, :]
            else:
                img_crop = img[y_min:y_max, x_min:x_max]
            img_crop = Image.fromarray(img_crop)
            img_crop.save(tgt)
        else:
            tgt = path + "train_set/" + model + "/o_" + file
            shutil.copyfile(img, tgt)

    # récup images
    org = src + "car_models_database_train/"
    l_img = os.listdir(org)
    for file in l_img:
        img = org + file
        if os.path.isfile(img):
            model = file.split("_")[0]
            img = org + file
            tgt = path + "train_set/" + model + "/" + file
            shutil.copy(img, tgt)
