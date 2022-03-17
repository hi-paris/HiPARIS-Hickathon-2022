Hi!Gency

HI!CKATHON 2022

Date : 06/03/2022

# I.	OVERVIEW
## 1.	Project Background and Description
 	Describe how this project came about, and the purpose.

Our guiding idea was to come up with an innovative tool that could help raise people's awareness of the carbon footprint of their personal vehicles.  Indeed, within global carbon emissions, car emissions play a major role and it is of great importance to try to reduce this impact. Furthermore, there do not seem to be effective transmission channels for individuals to realise the $CO_2$ consumption of their vehicles.

This is the main reason why we decided to create the Daisy model, which combines notions of Computer Vision and Machine Learning that will be detailed in the following parts, which will determine an estimation of the carbon emission of a vehicle.

Once the model has been conceived and with the help of local authorities and partner companies, a set of display panels with cameras (manufactured by a company under the Daisy brand) will be installed in strategic spots of the cities and when a car passes, drivers will have an estimation of their carbon emission based on the vehicle they are in.

All this data can be stored, aggregated and used to build statistics and scores at region or country levels, in order to have a better understanding of the car emissions by area. This information is also valuable for local authorities, because from them they will be able to enrich their “green” reputation and move towards a smart and sustainable city.

Finally, from a financial point of view, the way in which this project can generate revenue is by selling the data collected to manufacturing companies (e.g. cars) that will find great value in this data (e.g. segmentation of their respective markets).


## 2.	Project Scope
 	 Scope answers questions including what will be done, what won’t be done, and what the result will look like.

Daisy is a project developed with a vision to help humanity take the next big step in sustainability. 
In its deployment, this project will:
- Deploy daisy stations in major cities around the country
- Define a daisy score that will be a human friendly metric to measure the carbon footprint of vehicles
- Encourage the purchases of green-er cars by its exclusive collaboration with green companies to display advertisements on its displays (recommending the right green car, based on the passing car (e.g. recommending an electric sedan to a passing sedan))
- Push companies into making more enviroment-friendly decisions by awarding them with high daisy scores and special certificates
- Create a link between car manufacturers and local authorities
- Although this project was developed at Hi!Paris, this project will indeed scale to other major countries at first, and internationally later, to have the greatest impact on the enironment

This project:
- Will not display irrelevant advirtisements on its boards
- Will not favor, at any point, model performances over frugality
- Will not harvest rare earth materials to develop its hardware products
- Will not accept partnerships with companies bringing harm to the environment
- Will not use non-renewable energy sources

## 3.	Presentation of the group
 	Include your specialization at school, etc.

| First name | Last name    | Year of studies & Profile | School                   | Skills                                                      | Roles/Tasks                                                  | Observations |
| ---------- | ------------ | ------------------------- | ------------------------ | ----------------------------------------------------------- | ------------------------------------------------------------ | ------------ |
| Laurine    | Burgard-Lotz | 2022 - M2 DS              | M2 DS (Télécom SudParis) | General ML, DL, NLP, CV                                     | Technical - Image classification, carbon prediction           | ??           |
| Carlos     | Cortés       | 2022 - M2 DS              | M2 DS                    | General ML, DL, CV, economy                                 | Technical & Business - Computer vision, Video, business plan | ??           |
| Riad       | El Otmani    | 2022 - M2 DS              | M2 DS (Télécom SudParis) | General ML, DL, CV, video editing,                          | Technical & Business - Computer vision, Video, business plan | ??           |
| Louis      | Grenioux     | 2022 - M2 DS              | M2 DS (Télécom SudParis) | General ML, DL, NLP, CV                                     | Technical - Box prediction, image classification             | ??        |
| Emilien    | Grillot      | 2022 - M2 DS              | M2 DS (Télécom SudParis) | General ML, DL, NLP, CV                                     | Technical - Box prediction, image clasiffication                             |     ??         |
| Karim      | Kassab       | 2022 - M2 DS              | M2 DS (Télécom Paris)    | General ML, DL, CV, video editing, filming, video animation | Technical & Business - Video, business plan                  | ??           |
| Walid      | Yassine      | 2022 - M2 DS              | M2 DS (Télécom Paris)    | General ML, DL, NLP, CV                                     | Technical - Box prediction                                   | ??           |




## 4.	Task Management
 	Describe how you interacted and collaborated as a team, and the effect of every member’s unique background on the project.

As we were all good friends from the M2 Data Science at École Polytechnique, we knew each other, and our strength and weaknesses. Quickly, Louis took the lead of the group. We organized ourselves by starting with a big brainstorming session and then scheduling different deadlines for the 2 days and having wrap-up meetings at each big step (about every 6 hours). At each meeting everyone presents their progress and discusses the progress of others.
We also wrote everything down on the board to keep track of our progress and to facilitate the writing of the report and the distribution of future tasks.

As we all come from a scientific background, we decided that we would all test to advance on a part of the pipeline on Friday night (car recognition, box prediction and carbon prediction). On saturday morning we defined a business plan all together. Then, since some of us also have knowledge of video editing, we split the tasks between technical and business, while continuing to interact with each other and provide regular updates. For each step of the pipeline, each of us tested a different approach, and the most successful approaches were explored further.



# II.	PROJECT MANAGEMENT
## 1.	Data Understanding
 	Provided the initial collection of data has already occurred, this step includes identifying and defining the relevant data, exploring the range, scale, formats, contents, and biases of the data, and evaluating the quality and validity of the resulting data.

Regarding the car images dataset, the data provided is mainly split into 3 parts:

- Images of cars, with a bounding box detecting accurately the car in the foreground
- Images of cars, but with a bounding box detecting another object in the image (umbrella, tree,...)
- Other images that are not related to cars, with a randomly assigned bounding box

This very wild and heterogenous data was a big challenge.

One major point that had to be taken in consideration is the images that have more than one car in them. Therefore, our model has to detect the car that is in the foreground (the main car).

Also, we have to take special consideration regarding the images that have cars in them but are annotated based on other objects. These are some confusing images that may affect the backpropagation when we are training the model.

Other points that are worth pointing out is that the majority of images are RGB (3 channels), but there are some images with 4 channels and others with 1 channel (grayscale images). To be able to fairly deal with such cases, in the case of 4 channels, we will clip the channels to only 3 and in the case of grayscale images we will stack them on 3 channels to create a grayscale image but with RGB channels.

All these points will be taken into consideration in the model development phase.


## 2.	Data Pre-processing
 	Explain how the selection of data was manipulated and modified to remove redundant features and improve the quality of the data. Describe the preprocessing techniques used, such as data augmentation.


To improve the quality of our data, we focused on 2 major points:
- Selecting the right images to use for training and fine-tuning to avoid having erroneous data and updating the parameters of our model based on false data. For this we made sure to remove the images which have cars in them but are not labeled as cars, from the dataset that we created to do the fine-tuning and/or training.
- The machine learning models we use are more efficient when they work with a large amount of data (training directly on large batches). In this perspective, we have artificially increased the size of our dataset by generating new images from the old ones by making them undergo different transformations: translations, rotations, etc. To go even further in this direction, we scrapped some images of cars online to further augment our dataset.

These different steps allow us to put our algorithms in the best conditions so that they can generalize well during the training phase and identify the car model even if the image has undergone changes (quality degradations, translations, rotations, etc.)



## 3.	Modeling Development
 	Describe how you selected algorithms, how you calibrated them according to the data and how - in fine - you selected the best AI model using a well-defined set of metrics.

We have divided our reasoning into three major steps.

* **1st step :** build a model that would indicate whether we were dealing with an image of a car or not;
* **2nd step :** build a model that would allow us to find the limits of the box we were looking for;
* **3rd step :** build a model that could predict the CO2 emission from the car model.

Step 0:
- As a first step, we tried to use a CNN model from scratch. However, this one was overfitting and we were stuck at 80% accuracy.
- Because we knew state-of-the-art models could achieve way better results, we approached the problem using the VGG-16 pre-trained weights. VGG16 is a widely used Convolutional Neural Network (CNN) Architecture used for ImageNet. We finetuned it to adapt it to a binary classification problem. One first intuition for the choice of this model was that it is pretrained for classification task on a huge dataset. Thus it is pretty good for frugality since it avoids unnecessary training. Moreover, the basic architecture with frozen weights augmented with trainable dense layers, and dropout layer to reduce overfitting, gave us 97% accuracy.

Step 1:

- Determinist approach using step 0: 
    - Knowing that the car was regularly in the center of the image, we tested several image segmentation algorithms like KMeans, Otsu and Hidden Markov Chain without success. Indeed, these algorithms paid too much attention to the colour of the objects, which did not allow us to highlight the foreground.

- Statiscal approach using step 0:
    - Naive approch for step 0
        - As we observed that most cars where centered in the image (like from a google image search) and thought about the following idea : just provide the `width` and `height` as input and `x_min`, `y_min`, `x_max` and `y_max`. This task can simply be done by a random forest which is very frugal and cheap. This experiment lead to a `74%` mean IOU which is bad compared to other models but with exceptionaly fast performance on both training and inference.
    - Another idea was to transform the image into a pixel string using a Peano path. This string was to be passed to the input of a BLSTM and for each output of the BLSTM a softmax layer was applied to predict whether the pixel represented the surface of the viewer or not. We were not able to exploit this solution further because we did not have a RAM asserter (in particular to calculate the Peano paths).
    
- Statistical approach bypassing step 0: Yolo
    - Yolo is a very good model that allows us to identify and detect object at the same time. Therefore, at the output of the Yolo model, we would have directly the output of the step 0 (the class) and the step 1 (the bounding box coordinates). Our motivation behind this is that we would perform 2 steps in less time and hence gain frugality. We mainly explored 2 types of Yolo architectures:
        - Full architecture: Yolov4, Yolov3
        By using the pre-trained models, we gain in frugality by avoid using resources to train the model from scratch. Yolov4 was the model with the best prediction accuracy and an acceptable inference time.
        - Tiny architecture: Yolov4-tiny
        As an attempt to further enhance the frugality, we decided to take a version of yolov4 with a tiny architecture (more simplified) and to fine tune-it on the train dataset that we have in order to boost its performance. Indeed, the fine tuning helped improved its performance to 79.53%  with an inference time of 100 ms / image, which is very good in comparision with other architectures. However, in terms of iou performance, it is not quite performant as the complete architecture.
        - Fine tuned Yolov4:
        We also explored the approach of fine tuning the yolov4 architecture, However, since it is quite big, it takes a lot of time to do the fine tuning and uses a lot of resources. We decided to not go further in this direction since it is very costly in terms of frugality.

- Statistical approach bypassing step 0: FastRCNN and SSD
        We also wanted to use models from the [Tensorflow Hub](https://www.tensorflow.org/hub?hl=fr) repository which is reputed for its wide variety of models. Here, we investigated a lot of models which were fast but highly inaccurate but finally settled on `FastRCNN` and `SSD` combined with well-known image classsifiers. Unfortunaely, they weren't as good as YOLOs and were very expensive (especially `FastRCNN`)

Our decision of using the yolov4 architecture in our model is the result of the comparison between all the models that we evoked, by taking in consideration the performance and the frugality.


Step 2:

- Car image to car model using features:
    - In order to better estimate the model of the car, and in order to reduce the variance of our CO2 emission estimates, we explored an approach where we group the similar models together (in terms of CO2 emission) and create some sort of "model bags". Identifying the bag to which each car belongs may help us in reducing the uncertainity of our decision. This method has been explored manually, and assessed on visualizations, the results were very good but due to time constraints we will leave its implementation in the pipeline to future improvements.
   

- Car model to CO2 emission:
    - In order to improve our prediction, we proceeded by performing a feature engineering to improve the quality of the input of our regressor. Indeed, from the model, we have informed which type of car it is (race car, van, etc.) and the brand. We were also able to verify that there was no correlation between the year of the model and its emission.
    - We have tried another way in order to provide a perfect embedding of the model. We encoded the image in a latent space, our thinking being based on the fact that the standing of a car is also visible in the background of the image. After building an autoencoder, we performed a principal component analysis and found gaps in the explanatory power to determine the associated model. We have concluded that the background of the image only noised the inference.

- Car image to car model  (ResNet50, EfficientNet)
    - The first model that came to mind when trying to associate an image to a $CO_2$ carbon emission was to first dectect the car model which was linked by a one-to-one relation with the carbon emissions. Thus, our main task for this model was to create a classification model with 100 classes on the output. Our first choice was to do transfer larning by using the well known architecture Resnet50 for which we kept all layers except the last one and replaced it by a softmax classification layer that we fine-tuned based on our datase For this architecture we obtained the a good performance in termns of frugality but the results on the MAPE were not promising. We tried to combine the different predictions by using the weights from the softmax to average the CO2 predictions from several cars and not only keeping the best prediction. That is why we decided to explore a more sophisticated approach.
    - The same reasoning was used on the EfficientNet architecture, to test if it would improve the MAPE metric of our model. The results improved a bit in terms of frugality.
    
- Car image to CO2 emission directly:
    - We felt like the model to classify the car model was not efficient enough to rely entirely on it for carbon prediction. We decided to explore a new approach based on regression methods rather than classification methods. The input is the image (after preprocessing it) and the output is directly the CO2 emission of the associated car model (found in the training dataframe). We finetuned the previous used EfficientNet model since it already gave us best results for classification task. Instead of adding a Dense layer with 100 units, we used a linear layer with 1 output, which is the CO2 emission itself. We actually achieved a MAPE equal to 0.28 on our validation set, which a great improvement of 0.1 compared to the previous method, with a similar training time. This seemed more coherent to us because this method allows us to group together cars that have a certain shape but do not necessarily share the same model, and yet produce a fairly similar amount of carbon. Actually, this method is the "deep learning way" of finding and using the features previoulsy mentionned.
    


Models benchmark:


|Step                        |                                          |                           |Train       |Train                               |Train                           |Validation              |Validation                               |Validation                             |
|-------------------------|--------------------------------------------|-------------------------------|--------|-------------------------------|----------------------------|--------------|-------------------------------|-----------------------------|
|                     |                                        |                           |**Accuracy**|**IOU**                           |**Frugality (train time/image)**|**Accuracy**      |**IOU**                            |**Frugality  (train time/image)**|
|**Step0 : Car classification**|                                            |`VGG-16 (Fine tuned)`           |100%       |                           |630 ms                      |96.6%         |                           |530 ms                       |
|**Step1: Car detection**     |                        |Naïve (Random Forests)         |    |0.7415                         |2 ms                        |          |0.7414                         |1 ms                         |
|**Step1: Car detection**     |                        |`Yolov4 (Pre-trained)`           |    |                           |`0 ms`                        |`F-score: 92%`  |`0.935`                 |`550 ms`                       |
|**Step1: Car detection**     |                        |Yolov3 (Pre-trained)           |    |                           |0 ms                        |F-score: 90.6%|0.891                         |400 ms                       |
|**Step1: Car detection**     |                       |Yolov4-tiny (Fine tuned)       |None    |                           |460 ms                      |F-score:  91% |0.7953                         |100 ms                       |
|**Step1: Car detection**     |                        |Yolov4 (Full arch - Fine tuned)|    |                           |1850 ms                     |F-score: 68%  |0.92                           |550 ms                       |
|
|**Step1: Car detection**     |                        |Fast RCNN + Inception v2|    |                           |0 ms                     |93.6%  | 0.406                           |31000 ms                       |
|
|**Step1: Car detection**     |                        |SSD + Mobilenet v2|    |                           |0 ms                     |75%  |0.40                           |275 ms                       |
|
|                                        |                           | |  |**MAPE (Direct) / MAPE (Weighted)**| **Frugality (train time/image)**|**None**          |**MAPE (Direct) / MAPE (Weighted)**|**Frugality  (train time/image)**|
|**Step2: Carbon estimation** |Case1: Classification of Model + Deducing e |Resnet50                       |    |0.399/0.381                    |400 ms                      |          |0.408/0.386                    |4 ms                         |
| **Step2: Carbon estimation** |                                            |Efficient Net (for classification)                 |    |0.454/0.383                    |332 ms                      |          |0.444/0.377                    |24 ms                        |
|**Step2: Carbon estimation**  |Regression Model                            |`Efficient Net (for regression)`                 |    |`0.25`                           |`423 ms `                     |          |`0.28`                           |`20 ms`                        |


## 4.	 Deployment Strategy
 	What best practices/norms did you follow? How do you plan on deploying your IA solution?
The deployment of our strategy is to connect the three models in a data transfer pipeline. For the hackathon, we did this by hand, but eventually we should deploy the entire pipeline in the cloud to enable automation and parallelization of the process as shown in the deployment sheet below.

![](https://i.imgur.com/TdvOOBT.png)
### Guidelines
* Microservice architecture 
    * Easy way to monitor each part of our solution
    * Simplified service maintenance
* Low consumption hardware
    * Low power consumption display
    * Software part hosted in the cloud rather than on hardware
* Scalable system to handle a large number of requests
    * Kafka as message broker


### Explanations
For the deployment of our solution, it seems important to us to be able to remotely control all the Daisy meters in order to monitor the different machine learning models. To do this, we opted for a microservice architecture using docker containers where each machine learning model operates in the cloud to process the data coming from the daisy meters.

To ensure the parallelization of the process, we use Kafka as a messaging agent which allows to transfer information between the different containers in an efficient and very scalable way thanks to the cluster management that Kafka offers.

The panels and cameras have only a few software parts in order to limit their $CO_2$ consumption. The cameras must send their images and the panels must be able to receive a response. For that we implement a light API with Flask which requires very few hardware resources.


# III.	CARBON FOOTPRINT LIMITATION
 	Describe the taken measures/actions during the development of your solution in view of limiting the carbon footprint.
It goes without saying that the reason we started this project was to reduce $CO_2$ emissions.

As far as our technical development is concerned, different measures were taken throughout the project to limit carbon emissions. 

At the time of our first meeting, we thought of possible techniques to solve the problem. This techniques were assessed, not only from the performance and technical point of view, but also from the $CO_2$ emission point of view. Among the techniques with the less carbon impact that we decided to explore (not necessarily implement) were the following:

* Deterministic methods: One of the first methods we came up with to accomplish different tasks (such as box placing) were deterministic methods. From an environmental point of view, this type of methods should be privileged, because they do not need any kind of "Machine Learning type" training which corresponds to the main source of carbon emissions in this processes.
* Use pretrained models: As said before, the most demanding part of a Machine Learning algorithm corresponds to the training. By using transfer learning methods, we can reduce this impact because many layers weights remain unchanged and do not require any training or a slight fine-tuning.
* Trade-off frugality / performance: During the whole process we paid particular attention to the frugality of the Machine Learning algorithms we used. We made decisions on which model to use by weighing the run time (displayed in the the <b>Modeling Development</b> part) and the efficiency of the models. A trade-off between flugarity and performance was always considered.


# IV.	CONCLUSION
 	Tell us about the actual results, their limitations as well as future perspectives and improvements.

Regarding the car detection, our model is performing quite good. 
However, we still can further explore the fine-tuning approach, by working on the hyper-parameters, learning rate, etc. As evoked before in the model development part, there are plenty of approaches that can help improve the performance of the model, especially in the carbon detection part (like the approach of embedding). 
We believe that our solution, from a technical as well as a business perspective, in addition to its humanitarian perspective, will help humanity take the next big step towards a more sustainable environment.



