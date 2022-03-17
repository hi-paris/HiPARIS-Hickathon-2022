from torch import nn
from torch.nn import Module
from torchvision.models import detection
from torchvision import transforms
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS = {
    "frcnn-resnet": detection.fasterrcnn_resnet50_fpn,
    "frcnn-mobilenet": detection.fasterrcnn_mobilenet_v3_large_320_fpn,
    "retinanet": detection.retinanet_resnet50_fpn,
}

class carDetector(Module):
    """
        Module that output an cars bbox given an image
        Empty list is there is no care
    """
    
    def __init__ (self, n_classes=91, car_idx=3, verbose=False, min_prob=0.5, scale=False):
        # Initialisation du modèle
        super().__init__()
        
        self.car_idx = car_idx
        self.min_prob = min_prob
        self.scale = scale
        
        ## Initialisation of the network
        self.retinanet = MODELS["retinanet"](pretrained=True, progress=verbose, num_classes=n_classes, pretrained_backbone=True)
        self.network = nn.Sequential(*[
            self.retinanet
        ])
        self.resizer = transforms.Lambda(lambda x: x/255.)
        self.network.eval()
        
    def predict(self, x):
        """
            Given a tensor of image return the car bbox
            
            Parameters:
            -----------
            x: tensor containing each image to predict
        """
        
        with torch.no_grad():
            if self.scale:
                x = self.resizer(x)
            predictions = self.network(x)

            # Filter the car
            car_mask = [(prediction["labels"] == self.car_idx) & (prediction["scores"] >= self.min_prob) for prediction in predictions]
            new_prediction = [dict([(x, y[mask].detach().cpu()) for x, y in prediction.items()]) for prediction, mask in zip(predictions, car_mask)]

            # Output the most propable class
            idx = [torch.argmax(prediction["scores"]).item() if prediction["scores"].shape[0] > 0 else None for prediction in new_prediction]
            bbox = [prediction["boxes"][box_id].numpy().tolist() if box_id is not None else [] for prediction, box_id in zip(new_prediction, idx)]
        
        return bbox