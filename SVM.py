import cv2
import numpy as np
from HOGExtractor import HOGExtractor
import pickle
import random
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class SVM:

    def __init__(self, positive_datapath, negative_datapath):

        self.positive_datapath = positive_datapath
        self.negative_datapath = negative_datapath
        self.hog_extractor = HOGExtractor(64, 9, 16, 2, True)
        self.svc = None
        self.scaler = None


    def train_svm(self):

        with open(self.positive_datapath, 'rb') as handle:
            positive = pickle.load(handle)

        with open(self.negative_datapath, 'rb') as handle:
            negative = pickle.load(handle)

        positive_data = np.asarray(positive[0])
        negative_data = np.asarray(negative[0])

        positive_label = positive[1]
        negative_label = negative[1]

        unscale = np.vstack((positive_data, negative_data)).astype(np.float64)
        scaler = StandardScaler().fit(unscale)

        X = scaler.transform(unscale)
        Y = np.asarray(positive_label + negative_label)

        train_data, test_data, train_label, test_label = train_test_split(X,
                                                                          Y,
                                                                          test_size=0.2,
                                                                          random_state=random.randint(1, 100))

        svc = SVC(gamma='scale')
        svc.fit(train_data, train_label)

        score = svc.score(test_data, test_label)
        print("Accuracy:" + str(score*100.0) + "%")

        self.svc = svc
        self.scaler = scaler


    def classify(self, image):

        feature = self.hog_extractor.get_features(image)

        scaled_feature = self.scaler.transform([feature])

        result = self.svc.predict(scaled_feature)

        print(result)

        return result


#svm = SVM("HogData/positive.pickle", "HogData/negative.pickle")
#svm.train_svm()
#image = cv2.imread("DD.jpg")
#svm.classify(image)





