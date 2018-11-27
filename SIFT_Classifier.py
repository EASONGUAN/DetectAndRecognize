import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from skimage.measure import ransac
from skimage.transform import ProjectiveTransform
from HOGExtractor import HOGExtractor



class SIFT:

    def __init__(self, class1, class2, class3, class4, types, ratio):
        # Initialize the four standard car models.
        self.class1_f = class1[0]
        self.class1_b = class1[1]

        self.class2_f = class2[0]
        self.class2_b = class2[1]

        self.class3_f = class3[0]
        self.class3_b = class3[1]

        self.class4_f = class4[0]
        self.class4_b = class4[1]

        self.types = types
        self.ratio = ratio


    def ransac_h(self, image1, image2):

        img1 = cv.imread(image1)
        img2 = cv.imread(image2)
        img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
        img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

        # Create sift instance
        sift = cv.xfeatures2d.SURF_create()
        # sift = cv.xfeatures2d.SIFT_create(contrastThreshold=0.03, edgeThreshold=10)
        # Get key-points and descriptors
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        # Get matches and sort by the distance
        bfm = cv.BFMatcher()
        matches = bfm.knnMatch(des1, des2, k=2)

        # Apply ratio test
        pass_ratio = []
        for x, y in matches:
            if x.distance < self.ratio * y.distance:
                pass_ratio.append(x)

        # If we have at least four pairs of matching points
        # We can calculate homography matrix
        if len(pass_ratio) > 4:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in pass_ratio])
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in pass_ratio])
            # Apply RANSAC to find the homography transformation matrix
            data, inliers = ransac((src_pts, dst_pts), ProjectiveTransform, min_samples=4,
                                   residual_threshold=2, max_trials=100)

            inliers = np.array(inliers)
            index = np.where(inliers == True)

            return len(inliers[index])

        # If we do not have enough points to calculate points
        else:
            return 0


    def get_matches(self, image1, image2, ratio):
        img1 = cv.imread(image1)
        img2 = cv.imread(image2)

        # Create sift instance
        # sift = cv.xfeatures2d.SIFT_create(contrastThreshold=0.03, edgeThreshold=10)
        sift = cv.xfeatures2d.SURF_create()
        # Get key-points and descriptors

        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        # Get matches
        bfm = cv.BFMatcher()
        matches = bfm.knnMatch(des1, des2, k=2)
        # Apply ratio test
        pass_ratio = []
        for x, y in matches:
            if x.distance < ratio * y.distance:
                pass_ratio.append([x])

        # Visualize the matches
        matched_image = cv.drawMatchesKnn(img1, kp1, img2, kp2, pass_ratio[:50],
                                          flags=2, outImg=None)
        return matched_image


    def classify(self, image):
        num_inliers = []

        num_inliers.append(self.ransac_h(image, self.class1_f) + self.ransac_h(image, self.class1_b))
        num_inliers.append(self.ransac_h(image, self.class2_f) + self.ransac_h(image, self.class2_b))
        num_inliers.append(self.ransac_h(image, self.class3_f) + self.ransac_h(image, self.class3_b))
        num_inliers.append(self.ransac_h(image, self.class4_f) + self.ransac_h(image, self.class4_b))

        num_inliers = np.array(num_inliers)
        index = np.argmax(num_inliers)

        print(num_inliers)

        return types[index]


if __name__ == '__main__':

    class1 = ['./siftTest/1-f.jpg', './siftTest/1-b.jpg']
    class2 = ['./siftTest/5-f.jpg', './siftTest/5-b.jpg']
    class3 = ['./siftTest/3-f.jpg', './siftTest/3-b.jpg']
    class4 = ['./siftTest/6-f.jpg', './siftTest/6-b.jpg']
    types = ['Audi A5', 'Jeep Wrangler', 'BMW 2-SERIES', 'MERCEDES-BENZ CLA-CLASS']

    ratio = 0.9
    image = 'detected_car.jpg'

    sift =  SIFT(class1, class2, class3, class4, types, ratio)

    # plt.imshow(sift.get_matches(image, class2[1], ratio))
    # plt.show()

    print(sift.classify(image))