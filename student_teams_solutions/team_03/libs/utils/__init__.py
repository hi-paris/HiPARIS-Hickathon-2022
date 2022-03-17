from PIL import Image, ImageDraw
import numpy as np

import torch

def draw_img_boxes(image, boxes):
    """
        Return an image with a drawn label

        Parameters:
        -----------
        image: image numpy array in format (n_layer, h, w) and type uint8
        bbox: list of bbox in format xmin, ymin, xmax, ymax

        Output:
        -------
        image: numpy array in format (n_layer, h, w)
    """

    # Loading image with PIL
    image = Image.fromarray(image)
    image_draw = ImageDraw.ImageDraw(image)

    # Drawing box
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        image_draw.rectangle(((xmin, ymin), (xmax, ymax)), outline=(0, 0, 0))

    image.show()


def crop_imgs(imgs, boxes, transform):

    cropped_imgs = []
    for i in range(boxes.shape[0]):
        #print(int(boxes[i,1].item()),int(boxes[i,3].item()), int(boxes[i,0].item()),int(boxes[i,2].item()))
        img = imgs[i, :, int(boxes[i,1].item()):int(boxes[i,3].item()), int(boxes[i,0].item()):int(boxes[i,2].item())]
        #print(img.shape, i)
        img = transform(img)
        cropped_imgs.append(img.unsqueeze(0))

    return torch.cat(cropped_imgs, dim=0)

def imageToTensor(x):
    """
        Parameters:
        -----------
        x: numpy array of size (h, w, 3)
        
        Output:
        -------
        Float tensor of size (3, h, w)
    """
    
    x_ = torch.tensor(x, dtype=torch.float32)
    
    if x_.dim() == 3:
        x_ = torch.moveaxis(x_, 2, 0)
    
    return x_

def box_dataset_generator(box_dataset, output_folder, car_detector, device):
    """
        Generate cropped images according to car detector prediction
        Write them in a dedicated folder
        
        Input:
        -----
        box_dataset: box_dataset object
        output_folder: folder where to write files
        car_detector: instance of the car detector
        device: device for execution of the car_detector
        
        Output:
        -------
        X: list of filepaths
        y: list of labels
    """
    
    car_detector = car_detector.to(device)
    
    X = []
    y = []

    for x, i in zip(box_dataset, range(len(box_dataset))):
        x0, x1 = x
        x0 = x0.to(device)

        # Getting bbox
        bbox = car_detector.predict(x0)[0]

        # Creating image
        if len(bbox) == 1:
            image_path = f"{output_folder}/{i}.jpg"
            
            bbox = [int(x) for x in bbox[0]]
            image = torch.moveaxis(
                x0[0][:, bbox[1]:bbox[3], bbox[0]:bbox[2]],
                0,
                2
            ).detach().cpu().numpy()
            image = (image*255).astype(np.uint8)

            Image.fromarray(image).save(image_path)
            
            X.append(image_path)
            y.append(x1.item())
            
    return X, y