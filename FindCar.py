import Helper
from HOGExtractor import HOGExtractor
from SVM import SVM
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

class CarFinder:
    def __init__(self, hogExtractor, svm, windowSize, slide_step):
        # self.windowSize = windowSize
        self.slide_step = slide_step
        # self.low_thresh = thresh_low
        # self.high_thresh = thresh_high

        self.svm = svm

        self.hogExtractor = hogExtractor


    def find_car(self, file_name):

        img = cv.imread(file_name)

        self.windowSizeX = int (img.shape[1] / 3)
        self.windowSizeY = int (img.shape[0] / 3)
        valid_windows = []
        valid_image= []
        break_sig = 0
        for bottom_left_x in range(self.windowSizeX, img.shape[1], int(img.shape[1] / 6)):
            for bottom_left_y in range(self.windowSizeY, img.shape[0], int(img.shape[0] / 6)):

                x = (bottom_left_x - self.windowSizeX, bottom_left_x)
                y = (bottom_left_y - self.windowSizeY, bottom_left_y)

                if self.window_classfy(x, y, img) == 1:
                    valid_windows.append((x, y))
                    # only get the first image
                    img = np.copy(img)[y[0]:y[1], x[0]:x[1], :]
                    print("OK")
                    plt.imshow(img)
                    plt.show()
                    #cv.imwrite('test1.jpg', img)
                    #break_sig = 1
                    #break

            if break_sig == 1:
                break

    def window_classfy(self, x, y, img):

        img = np.copy(img)[y[0]:y[1], x[0]:x[1], :]

        if svm.classify(img) == ['positive']:
            return 1
        else:
            return 0


svm = SVM("HogData/cars.pickle", "HogData/negative_far.pickle")
svm.train_svm()
extractor = HOGExtractor(64, 12, 8, 2, True)

carFinder = CarFinder(extractor,svm, 64, 16)
carFinder.find_car('testC.jpg')

