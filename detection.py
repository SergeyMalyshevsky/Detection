import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_label_values_from_file(filename):
    with open(filename, 'r') as f:
        labels = f.readlines()

    coco_labels_dict = {}
    for label in labels:
        key, name = label.strip().split(':')
        coco_labels_dict[int(key)] = name.strip()
    return coco_labels_dict


def plot_preds(numpy_img, preds):
    filename = './static/labels/labels.txt'
    label_names = get_label_values_from_file(filename)

    boxes = preds['boxes'].detach().numpy()
    labels = list(preds['labels'])

    for i, box in enumerate(boxes, start=0):
        current_label = int(labels[i])
        numpy_img = cv2.rectangle(
            numpy_img,
            (box[0], box[1]),
            (box[2], box[3]),
            (0, 255, 0),
            1
        )
        cv2.putText(
            numpy_img,
            label_names[current_label],
            (int(box[0]), int(box[1]) - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (0, 255, 0),
            1
        )

    return numpy_img.get()


def detect_people(filename):
    input_folder = './static/uploads/'
    output_folder = './static/output/'

    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, num_classes=91)

    img_path = input_folder + filename
    img = cv2.imread(img_path)[:, :, ::-1]
    fig = plt.figure(figsize=(10, 5))

    model = model.eval()
    img_numpy = cv2.imread(img_path)[:, :, ::-1]
    img = torch.from_numpy(img_numpy.astype('float32')).permute(2, 0, 1)
    img = img / 255.

    predictions = model(img[None, ...])

    CONF_THRESH = 0.5
    boxes = predictions[0]['boxes'][predictions[0]['scores'] > CONF_THRESH]
    labels = predictions[0]['labels'][predictions[0]['scores'] > CONF_THRESH]

    boxes_dict = {}
    boxes_dict['boxes'] = boxes
    boxes_dict['labels'] = labels

    img_with_boxes = plot_preds(img_numpy, boxes_dict)
    fig = plt.figure(figsize=(10, 5))

    # save the image
    plt.imsave(output_folder + filename, img_with_boxes)


if __name__ == '__main__':
    filename = 'image.jpg'
    detect_people(filename)
