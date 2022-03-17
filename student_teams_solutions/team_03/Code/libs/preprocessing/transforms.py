import copy
import torch
from torchvision.transforms import RandomCrop
from torchvision.transforms import functional as F

class RandomCropWithBox(RandomCrop):
    def forward(self, input_value):
        """
            Return a cropped image and the new bounding box location
            
            Parameters:
            -----------
            input_value: tuple containing :
                img : the image tensor (n_channel, h, w)
                bbox : the list bounding box, in format xmin, ymin, xmax, ymax, tensor of size (n, 4)
                
            Output:
            -------
            Tuple :
            - cropped image of size (n_channel, h, w)
            - new bbox in format xmin, ymin, xmax, ymax
        """
        
        img, bbox = input_value
        bbox_copy = copy.deepcopy(bbox)
        
        if isinstance(bbox_copy, list):
            bbox_copy = torch.tensor(bbox_copy)
        
        # Getting the coordonates
        i, j, h, w = self.get_params(img, self.size)
        
        # Image preprocessing [copied from pytorch documentation]
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)

        width, height = F.get_image_size(img)
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img = F.pad(img, padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img = F.pad(img, padding, self.fill, self.padding_mode)
            
        # Getting the image
        image = F.crop(img, i, j, h, w)
        
        # Fixing the bounding box
        bbox_copy[:, 0], bbox_copy[:, 2] = bbox_copy[:, 0]-j, bbox_copy[:, 2]-j
        bbox_copy[:, 1], bbox_copy[:, 3] = bbox_copy[:, 1]-i, bbox_copy[:, 3]-i
        bbox_copy[bbox_copy <= 0] = 0
        bbox_copy[:, 2][bbox_copy[:, 2] >= image.shape[1]] = image.shape[2]
        bbox_copy[:, 3][bbox_copy[:, 3] >= image.shape[0]] = image.shape[1]
        bbox_copy = bbox_copy[(bbox_copy[:, 2]-bbox_copy[:, 0])*(bbox_copy[:, 3]-bbox_copy[:, 1]) != 0]
                                            
        return image, bbox_copy