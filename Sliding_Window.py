from HOG_Processor import HOGExtractor

import numpy as np
import cv2 as cv
import pickle
import matplotlib.pyplot as plt

class CarFinder:


    def __init__(self, hogExtractor, svm, windowSize, slide_step):
    	
        self.slide_step = slide_step
        self.svm = svm
        self.hogExtractor = hogExtractor
        self.windowSize = windowSize
        self.detected_window = None
        self.detected_car = None


    def find_car(self, file_name, mode):

        img = cv.imread(file_name)

        windowSizeX = self.windowSize
        windowSizeY = self.windowSize

        valid_windows = []
        valid_image= []

        for bottom_left_x in range(windowSizeX, img.shape[1], self.slide_step):
            for bottom_left_y in range(windowSizeY, img.shape[0], self.slide_step):

                x = (bottom_left_x - windowSizeX, bottom_left_x)
                y = (bottom_left_y - windowSizeY, bottom_left_y)

                if self.window_classfy( x, y, img, mode):
                    valid_windows.append([x[0], x[1], y[0], y[1]])
                    box_img = np.copy(img)[y[0]:y[1], x[0]:x[1], :]
                    valid_image.append(box_img)

        self.detected_window = valid_windows

        with open("HogData/validWindows_" + mode + '.pickle', 'wb') as handle:
            pickle.dump(valid_windows, handle, protocol=pickle.HIGHEST_PROTOCOL)

        num = 0
        for i in valid_image:
            cv.imwrite('results/' + str(num) + '.jpg', i)
            num += 1

        print(len(valid_windows))
        print('finished')


    def window_classfy(self, x, y, img, mode):

        img = np.copy(img)[y[0]:y[1], x[0]:x[1], :]

        return self.svm.classify(img, mode)


    def find_group(self, file_name, data_file):

        img = cv.imread(file_name)

        with open(data_file, 'rb') as handle:
            validWindows = pickle.load(handle)

        clusters = []
        for rect in validWindows:
            matched = 0
            for cluster in clusters:
                if (rect[0] <= cluster[1] and cluster[0] <= rect[1]
                    and rect[2] <= cluster[3] and cluster[2] <= rect[3]):
                    matched = 1
                    cluster[0] = min(cluster[0], rect[0])
                    cluster[1] = max(cluster[1], rect[1])
                    cluster[2] = min(cluster[2], rect[2])
                    cluster[3] = max(cluster[3], rect[3])

            if not matched:
                clusters.append(rect)

        detected = []
        for cluster in clusters:
            image = np.copy(img)[cluster[2]:cluster[3], cluster[0]:cluster[1],:]
            detected.append(image)
            cv.imwrite("haha.jpg",np.array(image))

        return clusters, detected

