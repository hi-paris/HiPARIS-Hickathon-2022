import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_output_layers(net):

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def extract_car(im_path, yolo_path, verbose=True):
    """
    Extract the box and the image of car delimited by the box if it is detected

    Parameters
    ----------
    im_path : path to image (.jpg)
    verbose : boolean, display or not the image

    Returns
    -------
    list_boxes : list of one bounding box if it is detected, empty list otherwise
    list_img : list of the delimitation of the image with the bounding box if it is detected, empty list otherwise
    """

    image = cv2.imread(im_path)
    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392

    classes = None

    with open(yolo_path + "yolov3.txt", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
        label = str(classes[class_id])
        color = COLORS[class_id]
        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
        cv2.putText(
            img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )

    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
    net = cv2.dnn.readNet(yolo_path + "yolov3.weights", yolo_path + "yolov3.cfg")
    blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    for out in outs:
        for detection in out:
            scores = detection[5:]
            # class_id = np.argmax(scores)
            class_id = 2
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    size_boxes = np.zeros(len(boxes))
    for i in indices:
        size_boxes[i] = boxes[i][2] * boxes[i][3]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]

    if len(boxes) > 0:
        ind_area_max = np.argmax(size_boxes)
        box = boxes[ind_area_max]
        x = box[0]
        y = box[1]
        x_left = max(x, 0)
        y_left = max(y, 0)
        w = box[2]
        h = box[3]
        list_img = [image[round(y) : round(y + h), round(x) : round(x + w)]]
        list_boxes = [box]
        if verbose:
            draw_prediction(
                image,
                class_ids[ind_area_max],
                confidences[ind_area_max],
                round(x),
                round(y),
                round(x + w),
                round(y + h),
            )
    else:
        list_boxes = []
        list_img = []

    if verbose:
        plt.imshow(image)
        cv2.waitKey()

    return list_boxes, list_img
