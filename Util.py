import cv2 as cv
import numpy as np
from HOG_Processor import HOGExtractor
import os
import pickle

def process_dataset(self, data_label, data_path):

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

    with open("HogData/" + "SIX" + '.pickle', 'wb') as handle:

        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)


def draw_box():

	#TODO


def draw_label():

	#TODO

def 