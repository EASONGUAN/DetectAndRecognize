import numpy as np
import cv2 as cv
import pickle
import matplotlib.pyplot as plt

from HOG_Processor import HOGExtractor
from Sliding_Window import CarFinder
from SIFT_Classifier import SIFT
from SVM_Classifier import SVM
from MLP_Classifier import MLP
from Util import process_dataset, draw_box, draw_label
from CNN_Classifier import CNN

import pickle
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras import callbacks

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Flatten, Dense
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, RMSprop


def main(image_file):
    # Process the dataset images, extract their hog features
    # process_dataset(1, "vehicle")
    # process_dataset(0, "non-vehicle")

    # Train SVM
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

    # Find vehicle fragmentation by the trained SVM
    with open("TrainedSvm/svm" + '.pickle', 'rb') as handle:
        svm = pickle.load(handle)

    extractor = HOGExtractor(64, 12, 8, 2, True)
    carFinder = CarFinder(extractor, svm, 125, 8)
    # carFinder.find_car(image_file, 'svc')
    # carFinder.find_car(image_file, 'linearSvc')

    # Combine the fragmentation into a whole vehicle image
    svc_clusters, svc_cars = carFinder.find_group(image_file, 'HogData/validWindows_svc.pickle')
    print(svc_clusters)
    linearSvc_clusters, linearSvc_cars = carFinder.find_group(image_file, 'HogData/validWindows_linearSvc.pickle')
    print(linearSvc_clusters)

    print(svc_clusters)
    print(linearSvc_clusters)

    # Visualize the valid windows and the detected clusters
    with open("HogData/validWindows_svc.pickle", 'rb') as handle:
        valid_windows_svc = pickle.load(handle)
    with open("HogData/validWindows_linearSvc.pickle", 'rb') as handle:
        valid_windows_linearSvc = pickle.load(handle)

    valid_windows_svc = draw_box(image_file, valid_windows_svc)
    valid_windows_linearSvc = draw_box(image_file, valid_windows_linearSvc)
    cv.imwrite('results/valid_windows_svc.jpg', valid_windows_svc)
    cv.imwrite('results/valid_windows_linearSvc.jpg', valid_windows_linearSvc)

    detected_svc = draw_box(image_file, svc_clusters)
    detected_linearSvc = draw_box(image_file, linearSvc_clusters)
    print(linearSvc_clusters)
    print(svc_clusters)
    cv.imwrite('results/detected_svc.jpg', detected_svc)
    cv.imwrite('results/detected_linearSvc.jpg', detected_linearSvc)

    for i in range(len(linearSvc_cars)):
        test_image = image.img_to_array(cv.resize(linearSvc_cars[i], (64, 64)))
        test_image = np.expand_dims(test_image, axis=0)
        tag = "haha"

        detected_linearSvc = draw_label(detected_linearSvc, tag, linearSvc_clusters[i])

    for j in range(len(svc_cars)):
        test_image = image.img_to_array(cv.resize(svc_cars[j], (64, 64)))
        test_image = np.expand_dims(test_image, axis=0)
        tag = "lol"
        detected_svc = draw_label(detected_svc, tag, svc_clusters[j])

    cv.imwrite("resultOOO.jpg", detected_linearSvc)
    cv.imwrite("resultJJJ.jpg", detected_svc)

    #classify the detected images


if __name__ == '__main__':
    main('bbb.jpg')

