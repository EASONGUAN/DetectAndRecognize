import numpy as np
import cv2 as cv
import pickle
import matplotlib.pyplot as plt

from HOG_Processor import HOGExtractor
from Sliding_Window import CarFinder
from SIFT_Classifier import SIFT
from SVM_Classifier import SVM
from MLP_Classifier import MLP
from Util import process_dataset

def main(image_file):
    # # Process the dataset images, extract their hog features
    # process_dataset(1, "vehicle")
    # process_dataset(0, "non-vehicle")
    #
    # # Train SVM
    # positive_datapaths = ['HogData/positive.pickle']
    # negative_datapaths = ['HogData/negative.pickle']
    #
    # svm = SVM(positive_datapaths, negative_datapaths)
    # svm.train_svc()
    # svm.train_linearSvc()
    #
    # # Store the trained SVM
    # with open("TrainedSvm/svm" + '.pickle', 'wb') as handle:
    #     pickle.dump(svm, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #
    # Find vehicle fragmentation by the trained SVM
    with open("TrainedSvm/svm" + '.pickle', 'rb') as handle:
        svm = pickle.load(handle)

    extractor = HOGExtractor(64, 12, 8, 2, True)
    carFinder = CarFinder(extractor, svm, 100, 16)
    carFinder.find_car(image_file, 'svc')
    carFinder.find_car(image_file, 'linearSvc')

    # Combine the fragmentation into a whole vehicle image
    svc_clusters = carFinder.find_group(image_file, 'HogData/validWindows_svc.pickle')


if __name__ == '__main__':
    main('./Visualize/1.jpeg')

