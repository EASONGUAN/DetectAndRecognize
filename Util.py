import cv2 as cv
import numpy as np
from HOG_Processor import HOGExtractor
import os
import pickle

def process_dataset(data_label, data_path):
    """

    :param data_label: "positive" or "negative
    :param data_path: path to the training data for SVM
    """
    files = os.listdir(data_path)

    output_data = []
    output_label = []

    for image_path in files:

        extractor = HOGExtractor(64, 12, 8, 2, True)

        real_path = data_path + "/" + image_path

        image = cv.imread(real_path)

        features = extractor.get_features(image)

        output_data.append(list(features))

        output_label.append(data_label)


    output = (output_data, output_label)

    name = "positive" if data_label == 1 else "negative"

    with open("HogData/" + name + '.pickle', 'wb') as handle:

        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)


def draw_box(image_file, valid_windows):
    """

    :param image_file: input image path
    :param valid_windows: array of windows
    :return: input image with box drawn base on valid_windows
    """
    img = cv.imread(image_file)
    for window in valid_windows:
        cv.rectangle(img, (window[0], window[2]), (window[1], window[3]), (0, 0, 255), 2)
    return img


def draw_label(image, tag, cluster):
    """

    :param image: input image
    :param tag: tag classified based on CNN
    :param cluster: grouped windows based on sliding_window
    :return: image with correspond label drawn on the top of each cluster
    """
    cv.putText(image, tag, (cluster[0], cluster[2]), cv.FONT_HERSHEY_SIMPLEX, 1,
               (255, 255, 255), 2, cv.LINE_AA)

    return image