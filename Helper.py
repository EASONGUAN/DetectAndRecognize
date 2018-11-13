import cv2 as cv
import numpy as np
from HOGExtractor import HOGExtractor
import os
import pickle


class DataProcessor:

    def __init__(self, dataset_path, data_label):

        self.dataset_path = dataset_path
        self.data_label = data_label


    def process_dataset(self):

        files = os.listdir(self.dataset_path)

        output_data = []
        output_label = []

        for image_path in files:

            extractor = HOGExtractor(64, 9, 16, 2, True)

            real_path = self.dataset_path + "/" + image_path

            features = extractor.get_features(real_path)

            output_data.append(list(features))

            output_label.append(self.data_label)

        output = (output_data, output_label)

        with open("HogData/" + self.data_label + '.pickle', 'wb') as handle:

            pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)


#processor = DataProcessor("POSITIVE/subcar", "positive")
#processor.process_dataset()