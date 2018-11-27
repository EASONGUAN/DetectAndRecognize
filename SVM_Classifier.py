from HOGExtractor import HOGExtractor
import numpy as np
import pickle
import random
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class SVM:

    def __init__(self, positive_datapaths, negative_datapaths):
    	
        print(positive_datapaths)
        self.positive_datapath = positive_datapaths
        self.negative_datapath = negative_datapaths
        self.hog_extractor = HOGExtractor(64, 12, 8, 2, True)
        self.svc = None
        self.linearSvc = None
        self.scaler = None


    def train_svc(self):

        positive = [[],[]]
        negative = [[],[]]

        for path in self.positive_datapath:

            with open(path, 'rb') as handle:
                temp = pickle.load(handle)

            positive[0] += temp[0]
            positive[1] += temp[1]

        for path in self.negative_datapath:

            with open(path, 'rb') as handle:
                temp = pickle.load(handle)

            negative[0] += temp[0]
            negative[1] += temp[1]

        positive_data = np.asarray(positive[0])
        negative_data = np.asarray(negative[0])

        positive_label = positive[1]
        negative_label = negative[1]

        unscale = np.vstack((positive_data, negative_data)).astype(np.float64)
        scaler = StandardScaler().fit(unscale)

        X = scaler.transform(unscale)
        Y = np.asarray(positive_label + negative_label)

        train_data, test_data, train_label, test_label = train_test_split(X, Y, test_size=0.2, random_state=random.randint(1, 100))

        svc = SVC(gamma='scale')
        svc.fit(train_data, train_label)

        score = svc.score(test_data, test_label)
        print("Accuracy:" + str(score*100.0) + "%")

        self.svc = svc
        self.scaler = scaler

        return svc, scaler


    def train_linearSvc(self):

        positive = [[],[]]
        negative = [[],[]]

        for path in self.positive_datapath:

            with open(path, 'rb') as handle:
                temp = pickle.load(handle)

            positive[0] += temp[0]
            positive[1] += temp[1]

        for path in self.negative_datapath:

            with open(path, 'rb') as handle:
                temp = pickle.load(handle)

            negative[0] += temp[0]
            negative[1] += temp[1]

        positive_data = np.asarray(positive[0])
        negative_data = np.asarray(negative[0])

        positive_label = positive[1]
        negative_label = negative[1]

        unscale = np.vstack((positive_data, negative_data)).astype(np.float64)
        scaler = StandardScaler().fit(unscale)

        X = scaler.transform(unscale)
        Y = np.asarray(positive_label + negative_label)

        train_data, test_data, train_label, test_label = train_test_split(X, Y, test_size=0.2, random_state=random.randint(1, 100))

        linear_svc = LinearSVC()
        linear_svc.fit(train_data, train_label)

        score = linear_svc.score(test_data, test_label)
        print("Accuracy:" + str(score*100.0) + "%")

        self.linearSvc = linear_svc
        self.scaler = scaler

        return linearSvc, scaler


    def classify(self, image, mode):

        feature = self.hog_extractor.get_features(image)

        scaled_feature = self.scaler.transform([feature])

        if mode == 'svc':

        	result = self.svc.predict(scaled_feature)

        elif mode == 'linearSvc':

        	result = self.linearSvc.predict(scaled_feature)

        return result