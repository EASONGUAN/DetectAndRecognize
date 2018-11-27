from HOG_Processor import HOGExtractor

import numpy as np
import cv2 as cv
import pickle
import random

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class MLP:

    def __init__(self, class1_datapaths, class2_datapaths, class3_datapaths):
        self.class1_datapaths = class1_datapaths
        self.class2_datapaths = class2_datapaths
        self.class3_datapaths = class3_datapaths
        self.hog_extractor = HOGExtractor(64, 12, 8, 2, True)
        self.mlp = None
        self.scalar = None


    def train_mlp(self):
        class1 = [[],[]]
        class2 = [[],[]]
        class3 = [[], []]

        for path in self.class1_datapaths:

            with open(path, 'rb') as handle:
                temp = pickle.load(handle)

            class1[0] += temp[0]
            class1[1] += temp[1]

        for path in self.class2_datapaths:

            with open(path, 'rb') as handle:
                temp = pickle.load(handle)

            class2[0] += temp[0]
            class2[1] += temp[1]

        for path in self.class3_datapaths:

            with open(path, 'rb') as handle:
                temp = pickle.load(handle)

            class3[0] += temp[0]
            class3[1] += temp[1]

        class1_data = np.asarray(class1[0])
        class2_data = np.asarray(class2[0])
        class3_data = np.asarray(class3[0])

        class1_label = class1[1]
        class2_label = class2[1]
        class3_label = class3[1]

        X = np.vstack((class1_data, class2_data, class3_data)).astype(np.float64)

        Y = np.asarray(class1_label + class2_label + class3_label)

        train_data, test_data, train_label, test_label = train_test_split(X, Y, test_size=0.2, random_state=random.randint(1, 100))

        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes = (5, 2), random_state = 1)
        clf.fit(train_data, train_label)

        score = clf.score(test_data, test_label)
        print("Accuracy:" + str(score * 100.0) + "%")

        self.mlp = clf

        return clf

    def classify(self, image):

        feature = self.hog_extractor.get_features(image)

        result = self.mlp.predict([feature])

        print(result)

        return result


if __name__ == '__main__':

    class1_datapaths = ['HogData/acura_cl.pickle']
    class2_datapaths = ['HogData/acura_el.pickle']
    class3_datapaths = ['HogData/acura_lix.pickle']

    mlp = MLP(class1_datapaths, class2_datapaths, class3_datapaths)
    mlp.train_mlp()