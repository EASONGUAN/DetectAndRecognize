import numpy as np
import cv2 as cv
import pickle
import matplotlib.pyplot as plt

from HOG_Processor import HOGExtractor
from Sliding_Window import CarFinder
from SIFT_Classifier import SIFT
from SVM_Classifier import SVM
from MLP_Classifier import MLP
from Util import process_dataset, draw_box
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
    carFinder = CarFinder(extractor, svm, 125, 10)
    # carFinder.find_car(image_file, 'svc')
    # carFinder.find_car(image_file, 'linearSvc')

    # Combine the fragmentation into a whole vehicle image
    svc_clusters, svc_cars = carFinder.find_group(image_file, 'HogData/validWindows_svc.pickle')
    linearSvc_clusters, linearSvc_cars = carFinder.find_group(image_file, 'HogData/validWindows_linearSvc.pickle')

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
    cv.imwrite('results/detected_svc.jpg', detected_svc)
    cv.imwrite('results/detected_linearSvc.jpg', detected_linearSvc)

    # Visualize the valid windows and the detected clusters
    with open('classifier.pickle', 'rb') as handle:
        classifier = pickle.load(handle)

    #Start training CNN
    cnn = CNN("TRAIN2","TEST2")
    cnn.vgg = classifier

    test_image = image.load_img('haha.jpg', target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = classifier.predict(test_image)
    print(result)

    #classify the detected images


if __name__ == '__main__':
    main('random.JPG')

