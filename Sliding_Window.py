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


    def find_car(self, file_name, mode):

        img = cv.imread(file_name)

        # windowSizeX = int (img.shape[1] / 8)
        # windowSizeY = int (img.shape[0] / 8)
        windowSizeX = self.windowSize
        windowSizeY = self.windowSize

        valid_windows = []
        valid_image= []

        for bottom_left_x in range(windowSizeX, img.shape[1], self.slide_step): #int(img.shape[1] / 12
            for bottom_left_y in range(windowSizeY, img.shape[0], self.slide_step): #int(img.shape[1] / 12

                x = (bottom_left_x - windowSizeX, bottom_left_x)
                y = (bottom_left_y - windowSizeY, bottom_left_y)

                if self.window_classfy( x, y, img, mode):
                    valid_windows.append([x[0], x[1], y[0], y[1]])
                    # only get the first image
                    box_img = np.copy(img)[y[0]:y[1], x[0]:x[1], :]
                    valid_image.append(box_img)

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


    def find_group(self, original_img, data_file):
        with open(data_file, 'rb') as handle:
            validWindows = pickle.load(handle)
        print(validWindows)
        img = cv.imread(original_img)

        windowSizeX = int(img.shape[1] / 6)
        windowSizeY = int(img.shape[0] / 6)

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

            if (not matched):
                clusters.append(rect)

        print(len(clusters))
        clusters = clusters[0]
        result = np.copy(img)[clusters[2]:clusters[3], clusters[0]:clusters[1], :]
        # result = np.copy(img)[clusters[2] + int(windowSizeY / 2):clusters[3] - int(windowSizeY / 2),
        #          clusters[0] + int(windowSizeX / 2):clusters[1] - int(windowSizeX / 2), :]
        cv.imwrite('results2/' + original_img + '.jpg', result)


# # Train SVM
# positive_datapaths = ['HogData/ONE.pickle', 'HogData/TWO.pickle', 'HogData/FOUR.pickle',
#                       'HogData/FIVE.pickle']
# negative_datapaths = ['HogData/GTI.pickle', 'HogData/Extras.pickle']
#
# svm = SVM(positive_datapaths, negative_datapaths)
# svm.train_svm()
# with open("HogData/svm2" + '.pickle', 'wb') as handle:
#     pickle.dump(svm, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Find vehicle fragmentation
# image_file = '9.jpeg'
# with open("HogData/svm" + '.pickle', 'rb') as handle:
#     svm = pickle.load(handle)
# extractor = HOGExtractor(64, 12, 8, 2, True)
# carFinder = CarFinder(extractor,svm, 125, 16)
# carFinder.find_car(image_file)

# Combine the fragmentation into a whole vehicle image
# 越野车1， threshold = 10， e = 0.2， size = 50, 6, svm
# 甲壳虫，threshold = 4， e = 0.2， size = 50, 6, svm
# 两辆车，threshold = 10， e = 0.2， size = 100
# 越野车2，threshold = 4， e = 0.2， size = 100
# threshold = 2
# e = 0.2
# size = 50
# carFinder.find_center(image_file, 'HogData/validWindows.pickle', threshold, e, size)
# carFinder.find_group(image_file, 'HogData/validWindows.pickle')



