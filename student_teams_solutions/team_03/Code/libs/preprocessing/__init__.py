from collections.abc import Iterable

import pandas as pd

from sklearn import preprocessing

import torch
from torch.utils.data import Dataset
from torchvision import transforms

import numpy as np

from PIL import Image

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import os
from glob import glob

os.chdir('/home/jovyan/')

def get_working_dir():
    return os.getcwd()

"""
    Pre-processing scripts
"""

def create_car_dataset(footprint_path, car_path, label="brand"):
    """Function that create the working dataset according to the CSV files
        Input:
            footprint_path: str, path of the footprint train dataset
            car_path: str, path of the footprint car and object dataset
            label: str, type of label, actually only brand of the car
            is accepted
        Output:
            tuple containing :
                X: pandas dataframe of images paths
                y: pandas dataframe of images box and labels
    """
    # Loading dataframes
    footprint_df = pd.read_csv(footprint_path, sep=";")
    car_df = pd.read_csv(car_path, sep=",")
    car_df["models"] = car_df["models"].str.strip()

    # Checking label
    accepted_labels = ["brand"]
    if label not in accepted_labels:
        raise ValueError(f"{label} is not an accepted label")

    # Creating the dataframe
    df = pd.merge(
        car_df,
        footprint_df,
        left_on="models",
        right_on="models",
        how="left"
    ).reset_index(drop=True)

    X = df["im_name"]
    if label == "brand":
        y = df[["x_min", "y_min", "x_max", "y_max", "Brand", "models"]].rename(
            columns={"Brand": "brand"}
        ).dropna()
        X = X.loc[y.index]
    
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    
    return X, y

def create_image_dataset(image_path):
    """Function that create the working dataset according to car images
        Input:
            image_path: str, path of the footprint car and object dataset
        Output:
            tuple containing :
                X: pandas dataframe of images paths
                y: pandas dataframe of images box and labels
    """
    
    # Get image list
    image_list = glob(f"{image_path}/*.jpg")
    
    # Create X and y
    X = pd.Series(image_list).str.split("/").apply(lambda x: x[-1])
    y = pd.Series(image_list).str.split("/") \
        .apply(lambda x: x[-1]) \
        .str.split(".").apply(lambda x: x[0]) \
        .str.split("_").apply(lambda x: "_".join(x[0:-1])) \
        .reset_index() \
        .rename(columns={0:"label"}) \
        .drop(columns=["index"])
    
    return X, y

class ImageDataset(Dataset):
    """
        Dataset class
        Class to deliver dynamically images from a specified folder
    """

    def __init__(self, image_folder, footprint_path, car_path, num_pixels=None, min_max_scaling=False, with_label=True):
        """
            Parameters:
            ----------
            images_names : [str] list of images names
        """
        super(ImageDataset).__init__()

        footprint_df = pd.read_csv(footprint_path, sep=";")
        car_df = pd.read_csv(car_path, sep=",")
        car_df["models"] = car_df["models"].str.strip()

        models_co2 = footprint_df[['models', 'Average of CO2 (g per km)']]
        dataset_models = car_df.loc[car_df['class'] == 'car'][['im_name', 'models']]

        dataset_co2_df = pd.merge(
                models_co2,
                dataset_models,
                left_on="models",
                right_on="models",
                how="left"
            ).reset_index(drop=True)
        dataset_co2 = dataset_co2_df.drop('models', axis=1).to_numpy()

        le = preprocessing.LabelEncoder()
        le.fit(dataset_co2_df['models'].to_numpy())

        dataset_co2_df['models_id'] = le.transform(dataset_co2_df['models'].to_numpy())

        img_name_to_id = dict(zip(dataset_co2_df['im_name'].to_numpy(), dataset_co2_df['models_id'].to_numpy()))
        
        self.with_label = with_label
        self.num_pixels = num_pixels
        self.min_max_scaling = min_max_scaling

        self.images_paths = glob(image_folder + '/*.jpg')

        self.id_to_co2 = dict(zip(dataset_co2_df['models_id'].to_numpy(), dataset_co2_df['Average of CO2 (g per km)'].to_numpy()))
        
        if with_label:
            self.dataset_labels = []
            for img_path in self.images_paths:
                img_name = img_path.split('/')[-1]
                self.dataset_labels.append(img_name_to_id[img_name] if img_name in img_name_to_id else -1)

    def __str__(self):
        n_data = self.__len__()

        return f"Dataset of {n_data} images"

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idxs):

        if isinstance(idxs, int):
            idxs = [idxs]

        imgs = torch.empty((len(idxs), 3, self.num_pixels, self.num_pixels))
        
        if self.with_label:
            labels = torch.empty(len(idxs), dtype=torch.int)

        for i, idx in enumerate(idxs):
            img = Image.open(self.images_paths[idx])

            if self.num_pixels is not None:
                img = img.resize((self.num_pixels, self.num_pixels))

            img = torch.Tensor(np.array(img))
 
            # 4 channels and more support
            if img.dim() > 2 and img.shape[2] > 3:
                img = img[:,:,:3]

            # black and white pictures support
            if img.dim() == 2:
                img = torch.tile(img.unsqueeze(2), (1,1,3))

            img = torch.moveaxis(img, 2, 0)


            imgs[i] = img

            if self.with_label:
                labels[i] = self.dataset_labels[idx]

        if self.min_max_scaling:
            imgs = imgs / 255
        
        if self.with_label:
            return imgs, labels
        else:
            return imgs

    def show_image(self, idx):
        img = Image.open(self.images_paths[idx])

        if self.num_pixels is not None:
            img = img.resize((self.num_pixels, self.num_pixels))

        img.show()


class BoxDataset(Dataset):
    """
        Dataset class
        Class to generate a dataset containing pictures and their bounding box if they represent a car
    """

    def __init__(self, img_dataset, model, device):
        """
            Parameters:
            ----------
            img_dataset : [Dataset] images dataset
        """
        super(BoxDataset).__init__()

        if img_dataset.num_pixels is None:
            print('Careful ! Images are not scaled')

        self.img_dataset = img_dataset
        self.model = model
        self.device = device
        
    def __str__(self):
        n_data = self.__len__()

        return f"Dataset of {n_data} images"

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.img_dataset)

    def __getitem__(self, idxs):

        if isinstance(idxs, int):
            batch_dim = 1
        else:
            batch_dim = len(idxs)
            
        boxes = torch.empty((batch_dim, 4))
        imgs, labels = self.img_dataset[idxs]

        model = self.model.to(self.device)
        for i, img in enumerate(imgs):
            img = img.to(self.device)

            with torch.no_grad():
                box = model.predict([img])[0]

                if len(box) == 0:
                    box = torch.zeros(4)
                else:
                    box = torch.tensor(box)
                    
            boxes[i] = box

        if isinstance(idxs, int):
            imgs = imgs.squeeze(0)
            boxes = boxes.squeeze(0)
            labels = labels[0].item()

        return (imgs, boxes), labels
    
class boxDataset2(Dataset):
    def __init__(self, X, y, size=256, transform=True):
        super().__init__()

        # Defining transformers
        resizer = transforms.Resize(size=size)
        cropper = transforms.CenterCrop((size, size))
        flipper_1 = transforms.RandomHorizontalFlip()
        flipper_2 = transforms.RandomVerticalFlip()
        rotater = transforms.RandomRotation((-90, 90))

        self.labels = [i for i in y if i != -1]
        self.images = [i for i, j in zip(X, y) if j != -1]
        self.transform = transform
        self.transformer = transforms.Compose([
            resizer,
            cropper,
            flipper_1,
            flipper_2,
            rotater
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__ (self, idx):
        X = self.images[idx]
        y = self.labels[idx]
        
        X = Image.open(X)
        
        if self.transform:
            X = self.transformer(X)
            
        X = transforms.ToTensor()(X)
            
        return X, y