from torch.utils.data import Dataset
from collections.abc import Iterable
import numpy as np
from PIL import Image

class imageDataset(Dataset):
    """
        Dataset class
        Class to deliver dynamically images from a specified folder
    """

    def __init__(self, images_names, image_folder):
        """
            Parameters:
            ----------
            images_names : [str] list of images names
        """
        super().__init__()

        image_paths = [f"{image_folder}/{x}" for x in images_names]
        self.image_paths = dict(zip(range(len(image_paths)), image_paths))  # Dict of images id and path
        self.image_list = list(self.image_paths.keys())  # Useful for dataset splitting

    def __str__(self):
        n_data = self.__len__()

        return f"Dataset of {n_data} images"

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.image_paths.keys())

    def __get_image(self, image_path):
        image = np.array(
            Image.open(image_path)
        )

        return image

    def __getitem__(self, s):
        images = []

        # Getting indices range
        if isinstance(s, slice):
            start = s.start if s.start is not None else 0
            stop = s.stop if s.stop is not None else self.__len__()
            step = s.step if s.step is not None else 1

            indices = range(start, stop, step)
        elif isinstance(s, int):
            indices = [s]
        elif isinstance(s, Iterable):
            indices = s
        else:
            raise NotImplementedError

        for idx in indices:
            images.append(
                self.__get_image(self.image_paths[self.image_list[idx]])
            )

        return images

    def get_from_id(self,  id):

        return self.__get_image(self.image_paths[id])