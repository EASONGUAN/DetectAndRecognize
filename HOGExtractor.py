import cv2 as cv
import numpy as np
from skimage.feature import hog
import matplotlib.pyplot as plt

class HOGExtractor:

    def __init__(self, box_size, orientation, pixel_per_cell, cells_per_block, do_trans_sqrt):

        self.box_size = (box_size, box_size)
        self.orientation = orientation
        self.pixel_per_cell = (pixel_per_cell, pixel_per_cell)
        self.cells_per_block = (cells_per_block, cells_per_block)
        self.do_trans_sqrt = do_trans_sqrt

        self.hue_hog = None
        self.saturation_hog = None
        self.lightness_hog = None


    def HOG(self, image):

        image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

        hue = image[:,:,0]
        saturation = image[:,:,1]
        lightness = image[:,:,2]

        hue = cv.resize(hue, self.box_size, interpolation = cv.INTER_AREA)
        saturation = cv.resize(saturation, self.box_size, interpolation=cv.INTER_AREA)
        lightness = cv.resize(lightness, self.box_size, interpolation=cv.INTER_AREA)

        hue_features, hue_feature_image = hog(hue,
                                              orientations=self.orientation,
                                              pixels_per_cell=self.pixel_per_cell,
                                              cells_per_block=self.cells_per_block,
                                              transform_sqrt=self.do_trans_sqrt,
                                              visualise=True,
                                              feature_vector=False)

        saturation_features, saturation_feature_image = hog(saturation,
                                              orientations=self.orientation,
                                              pixels_per_cell=self.pixel_per_cell,
                                              cells_per_block=self.cells_per_block,
                                              transform_sqrt=self.do_trans_sqrt,
                                              visualise=True,
                                              feature_vector=False)

        lightness_features, lightness_feature_image = hog(lightness,
                                              orientations=self.orientation,
                                              pixels_per_cell=self.pixel_per_cell,
                                              cells_per_block=self.cells_per_block,
                                              transform_sqrt=self.do_trans_sqrt,
                                              visualise=True,
                                              feature_vector=False)

        output = {'hue_hog': np.reshape(hue_features, -1),
                  'saturation_hog': np.reshape(saturation_features, -1),
                  'lightness_hog': np.reshape(lightness_features, -1),
                  'hue_hog_img': lightness_feature_image,
                  'saturation_hog_img': saturation_feature_image,
                  'lightness_hog_img': lightness_feature_image}

        return output


    def get_features(self, image):

        feature = self.HOG(image)

        output = np.hstack((feature['hue_hog'],
                            feature['saturation_hog'],
                            feature['lightness_hog']))

        return output












