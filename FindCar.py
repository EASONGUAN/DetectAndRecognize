import Helper
from HOGExtractor import HOGExtractor
from SVM import SVM

import numpy as np
import cv2 as cv
import pickle
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

        self.windowSizeX = int (img.shape[1] / 6)
        self.windowSizeY = int (img.shape[0] / 6)
        valid_windows = []
        valid_image= []
        # break_sig = 0
        for bottom_left_x in range(self.windowSizeX, img.shape[1], 32): #int(img.shape[1] / 12
            for bottom_left_y in range(self.windowSizeY, img.shape[0], 32): #int(img.shape[1] / 12

                x = (bottom_left_x - self.windowSizeX, bottom_left_x)
                y = (bottom_left_y - self.windowSizeY, bottom_left_y)

                if self.window_classfy( x, y, img):
                    valid_windows.append([x[0], x[1], y[0], y[1]])
                    # only get the first image
                    box_img = np.copy(img)[y[0]:y[1], x[0]:x[1], :]
                    valid_image.append(box_img)
                    # plt.imshow(box_img)
                    # plt.show()
                    #break_sig = 1
                    #break
            # if break_sig == 1:
            #     break

        with open("HogData/validWindows" + '.pickle', 'wb') as handle:
            pickle.dump(valid_windows, handle, protocol=pickle.HIGHEST_PROTOCOL)

        num = 0
        for i in valid_image:
            cv.imwrite('results/' + str(num) + '.jpg', i)
            num += 1

        print(len(valid_windows))
        print('finished')


    def window_classfy(self, x, y, img):

        img = np.copy(img)[y[0]:y[1], x[0]:x[1], :]

        if svm.classify(img) == ['positive']:
            return 1
        else:
            return 0


    def find_center(self, original_img, data_file, threshold, e):
        with open(data_file, 'rb') as handle:
            validWindows = pickle.load(handle)

        final_boxes = cv.groupRectangles(validWindows, threshold,  e)
        print(final_boxes)

        img = cv.imread(original_img)

        num = 0
        for box in final_boxes[0]:

            box_img = np.copy(img)[box[2]:box[3] + 20, box[0]:box[1] + 20, :]

            cv.imwrite('results/' + str(num) + '_final.jpg', box_img)

            num += 1


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
image_file = 'test2.jpg'
with open("HogData/svm2" + '.pickle', 'rb') as handle:
    svm = pickle.load(handle)
extractor = HOGExtractor(64, 12, 8, 2, True)
carFinder = CarFinder(extractor,svm, 64, 16)
carFinder.find_car(image_file)

# Combine the fragmentation into a whole vehicle image
threshold = 5
e = 0.2
carFinder.find_center(image_file, 'HogData/validWindows.pickle', threshold, e)



