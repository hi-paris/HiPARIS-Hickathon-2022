# Imports of the packages

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import torchvision.models as models
from PIL import Image
import json


# Hyperparameters
# Transformations of the training data
### Random rotations, flips, ...
def create_data(train_images_dir, test_images_dir, footprints_dir):
    """
    Inputs:
    - The directories of the training set, test set and the csv with the footprints info

    Output:
    - A dictionary with useful data for the other steps, such as the loaders for the train set, test set and valid test,
        and the matches for the models of the cars

    """
    train_transforms = transforms.Compose(
        [
            transforms.Resize((244, 244)),
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    test_transforms = transforms.Compose(
        [
            transforms.Resize((244, 244)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    ### Loading data

    train_data = datasets.ImageFolder(train_images_dir, transform=train_transforms)
    test_data_intermediary = datasets.ImageFolder(
        test_images_dir, transform=test_transforms
    )

    len_test = int(0.9 * len(test_data_intermediary))
    len_validation = len(test_data_intermediary) - len_test
    test_data, valid_data = torch.utils.data.random_split(
        test_data_intermediary, [len_test, len_validation]
    )

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32, shuffle=True)

    ### Loading footprint data

    footprints = pd.read_csv(footprints_dir + "car_models_footprint.csv", sep=";")
    models_emissions = footprints.set_index("models")
    models_emissions = models_emissions[["Average of CO2 (g per km)"]].to_dict()[
        "Average of CO2 (g per km)"
    ]

    ### Tie the class indices to their names

    classes = os.listdir(train_images_dir)
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}

    dct = {
        "trainloader": trainloader,
        "testloader": testloader,
        "validloader": validloader,
        "models_emissions": models_emissions,
        "classes": classes,
        "class_to_idx": class_to_idx,
    }

    return dct


def training(
    trainloader,
    validloader,
    lr,
    nb_epochs,
    momentum,
    patience,
    threshold,
    print_every,
    class_to_idx,
    nb_car_models,
):
    """
    Inputs:
    - The train data trainloader and validation data validationloader as DataLoaders
    - All the hyperparameters for the network (lr, momentum, patience, threshold, print_every)
    - class_to_idx : The correspondence tab for the car models

    Output:

    """

    # The model

    ### Layers

    model = models.resnet34(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, nb_car_models)

    ### Compile

    criterion = nn.CrossEntropyLoss()
    metrics = ["accuracy"]
    optimizer = optim.SGD(model.parameters(), lr, momentum)
    lrscheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=patience, threshold=threshold, mode="max"
    )

    ### Implement a function for the validation pass

    def validation(model, criterion, validloader=validloader):
        valid_loss = 0
        accuracy = 0

        # change model to work with cuda
        model.to("cuda")

        # Iterate over data from validloader
        for ii, (images, labels) in enumerate(validloader):

            # Change images and labels to work with cuda
            images, labels = images.to("cuda"), labels.to("cuda")

            # Forward pass image though model for prediction
            output = model.forward(images)
            # Calculate loss
            valid_loss += criterion(output, labels).item()
            # Calculate probability
            ps = torch.exp(output)

            # Calculate accuracy
            equality = labels.data == ps.max(dim=1)[1]
            accuracy += equality.type(torch.FloatTensor).mean()

        return valid_loss, accuracy

    ### Does the training

    model.to("cuda")
    model.train()

    steps = 0

    for e in range(nb_epochs):

        running_loss = 0

        # Iterating over data to carry out training step
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1

            inputs, labels = inputs.to("cuda"), labels.to("cuda")

            # zeroing parameter gradients
            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Carrying out validation step
            if steps % print_every == 0:
                # setting model to evaluation mode during validation
                model.eval()

                # Gradients are turned off as no longer in training
                with torch.no_grad():
                    valid_loss, accuracy = validation(model, criterion, validloader)

                print(
                    f"No. epochs: {e+1}, \
                Training Loss: {round(running_loss/print_every,3)} \
                Valid Loss: {round(valid_loss/len(validloader),3)} \
                Valid Accuracy: {round(float(accuracy/len(validloader)),3)}"
                )

                # Turning training back on
                model.train()
                lrscheduler.step(accuracy * 100)

    # Creates the output

    training_output = {
        "state_dict": model.state_dict(),
        "model": model.fc,
        "class_to_idx": class_to_idx,
        "opt_state": optimizer.state_dict,
        "num_epochs": nb_epochs,
    }

    return training_output


# Si tu n'enchaînes pas training et prediction, il faut enregistrer le modèle entraîné quelque part :


def launch_training(
    trainloader,
    validloader,
    lr,
    nb_epochs,
    momentum,
    patience,
    threshold,
    print_every,
    class_to_idx,
    model_save_dir,
    nb_car_models,
):

    ### Save the training output (essentially the model)

    torch.save(
        training(
            trainloader=trainloader,
            validloader=validloader,
            lr=lr,
            nb_epochs=nb_epochs,
            momentum=momentum,
            patience=patience,
            threshold=threshold,
            print_every=print_every,
            class_to_idx=class_to_idx,
            nb_car_models=nb_car_models,
        ),
        model_save_dir,
    )


# Prediction part


def predict_emission(
    image, model_save_dir, topk, models_emissions, nb_car_models, classes
):
    """
    Inputs:
    - One image, given on a matrix format RGB
    - A model path, containing the model trained during the training part and saved somewhere
    - topk : An integer, the number of possible predictions we are willing to use to compute the average carbon emission

    Ouputs:
    - One integer: the average carbon emission associated to the car in the image (Average of CO2 (g per km))
    """

    ### Load the model (ie the training output)

    def load_checkpoint(filepath=model_save_dir):

        checkpoint = torch.load(filepath)
        model = models.resnet34(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, nb_car_models)
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        model.class_to_idx = checkpoint["class_to_idx"]

        return model

    ### Process the given image:

    def process_image(image):
        # Process a PIL image for use in a PyTorch model

        pil_im = Image.fromarray(image, "RGB")

        # Building image transform
        transform = transforms.Compose(
            [
                transforms.Resize((244, 244)),
                # transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        # Transforming image for use with network
        pil_tfd = transform(pil_im)

        # Converting to Numpy array
        array_im_tfd = np.array(pil_tfd)

        return array_im_tfd

    ###Predictions of the model of the car from the image

    def predict(image, model_path=model_save_dir, topk=topk):
        # Implement the code to predict the car model from an image array

        loaded_model = load_checkpoint(model_path).cpu()

        # Pre-processing image
        img = process_image(image)

        # Converting to torch tensor from Numpy array
        img_tensor = torch.from_numpy(img).type(torch.FloatTensor)

        # Adding dimension to image to comply with (B x C x W x H) input of model
        img_add_dim = img_tensor.unsqueeze(0)

        # Setting model to evaluation mode and turning off gradients
        loaded_model.eval()
        with torch.no_grad():
            # Running image through network
            output = loaded_model.forward(img_add_dim)

        probs_top = output.topk(topk)[0]
        predicted_top = output.topk(topk)[1]

        # Converting probabilities and outputs to lists
        conf = np.array(probs_top.cpu())[0]
        conf = conf / conf.sum()
        predicted = np.array(predicted_top.cpu())[0]

        return conf / conf.sum(), predicted

    ### Given an array of predictions for the car model with their percentage confidence,
    ### returns the weighted mean for the average carbon emission

    def emissions_from_image(
        image,
        model_path=model_save_dir,
        topk=topk,
        models_emissions=models_emissions,
        classes=classes,
    ):
        # From an image, the model and the number of allowed predictions, returns the average carbon emission

        conf, predicted = predict(image, model_path, topk=topk)
        predicted_classes = [classes[predicted[i]] for i in range(topk)]
        mean_emissions = 0
        for i in range(topk):
            mean_emissions += conf[i] * models_emissions[predicted_classes[i]]
        return mean_emissions

    return emissions_from_image(image)
