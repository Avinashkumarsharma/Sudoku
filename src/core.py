#!/usr/bin/env python
"""
Author : Avik
Date : 29th Dec 2016
Description : Extract Sodoku from the given image input.

"""
import cv2
import numpy as np
from sklearn.externals import joblib
from sklearn import datasets
from sklearn.svm import LinearSVC
from sklearn.svm import
from skimage.feature import hog
class Image:
    """ Contains the image for processing and basic operations on it """
    def __init__(self, path_to_image='../data/sudoku.jpg'):
        self.img = cv2.imread(path_to_image, 0)
        self.height, self.width = self.img.shape
        self.dataset = None
        if len(self.img.shape) == 2:
            self.channel = 0
        else:
            self.channel = self.img.shape[3]
            cv2.cvtColor(self.img, self.img, cv2.COLOR_BGR2GRAY)
        if self.height > 500:
            self.resize(0.5)
            self.height *= 0.5
            self.width *= 0.5
            self.height = int(self.height)
            self.width = int(self.width)
        self.download_data()
        self.features = np.array(self.dataset.data, 'int16')
        self.label = np.array(self.dataset.target, 'int')


    @staticmethod
    def display(image, name='Image', duration = 0, resize = True):
        print 'Showing image'
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.imshow(name, image)
        cv2.waitKey(duration)
        cv2.destroyAllWindows()

    def exp(self):
        print self.img[100, 100]

    def resize(self, ratio=0.5):
        self.img = cv2.resize(self.img, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)

    def threshold(self):
        _img = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)
        self.img = cv2.bilateralFilter(self.img, 11, 17, 17)
        self.img = cv2.adaptiveThreshold(self.img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 5)
        _img = self.img
        self.img = cv2.Canny(self.img, 100, 255)
        (cnts, _) = cv2.findContours(self.img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
        grid = None
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02*peri, True)

            if len(approx) == 4:
                grid = approx
                break
        pt1 = np.float32([grid[0][0], grid[1][0], grid[2][0], grid[3][0]])
        pt2 = np.float32([[0, 0], [0, self.height], [self.width, self.height], [self.width, 0]])
        M = cv2.getPerspectiveTransform(pt1, pt2)
        self.img = cv2.warpPerspective(self.img, M, (self.width, self.height))
        _img = cv2.warpPerspective(_img, M, (self.width, self.height))
        cv2.drawContours(_img, [grid], -1, (0, 255, 0), 3)
        y = 6
        x = 4


        block = self.img[(y-1)*self.height/9:y*self.height/9, (x-1)*self.width/9:x*self.width/9]
        _img_block = _img[(y-1)*self.height/9:y*self.height/9, (x-1)*self.width/9:x*self.width/9]
        (digit, _) = cv2.findContours(block, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        digit = sorted(digit, key=cv2.contourArea, reverse=True)[0]
        x, y, w, h = cv2.boundingRect(digit)
        cv2.rectangle(_img_block, (x, y), (x+w, y+h), (0, 255, 0))
        self.display(_img_block)
        _digit = _img_block[y:y+h, x:x+w]
        #_invert = cv2.dilate(_digit, (3, 3))
        _invert = cv2.bitwise_not(_digit)
        _invert = cv2.resize(_invert, (28, 28), interpolation=cv2.INTER_AREA)
        self.display(_invert)
        return _invert

    def download_data(self):
        self.dataset = datasets.fetch_mldata('MNIST Original', data_home='../data/')

    def train(self):
        list_hog_fd = []
        for feature in self.features:
            fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14),
                     cells_per_block=(1,1), visualise=False )
            list_hog_fd.append(fd)
        hog_feature = np.array(list_hog_fd, 'float64')
        clf = LinearSVC()
        clf.fit(hog_feature, self.label)
        joblib.dump(clf, '../data/digits_clf.pkl', compress=3)

    def test(self):
        img = self.threshold()
        clf = joblib.load('../data/digits_clf.pkl')
        fd = hog(img, orientations=9, pixels_per_cell=(14, 14),
                 cells_per_block=(1, 1), visualise=False)
        nbr = clf.predict(np.array([fd], 'float64'))
        print int(nbr[0])





sudoku = Image('../data/sudoku_3.png')
#sudoku.train()
sudoku.test()
#sudoku.download_data()
#sudoku.threshold()
#Image.display(sudoku.img)

