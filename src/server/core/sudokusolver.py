#!/usr/bin/env python
"""
Author: Avik
Date: 30th Dec 2016
Description : Solver for sudoku from image

"""

import cv2
import numpy as np
from sklearn.externals import joblib
from sklearn import datasets
from sklearn.svm import LinearSVC
from skimage.feature import hog

class Sudoku:
    """
    Main sudoku solver class. Needs sudoku image to initialize

    """

    FIXED_IMAGE_HEIGHT = 500

    @staticmethod
    def display(image, name='Image', duration=0, resize=True):
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.imshow(name, image)
        cv2.waitKey(duration)
        cv2.destroyAllWindows()

    def __init__(self, image):
        image_byte = image.read()
        self.sudoku_img = cv2.imdecode(np.fromstring(image_byte, np.uint8), cv2.CV_LOAD_IMAGE_GRAYSCALE)
        self.height, self.width = self.sudoku_img.shape
        self.sudoku_img = self._preprocess()

    def _preprocess(self):
        if self.height > 700:
            resize_ratio = float(self.FIXED_IMAGE_HEIGHT)/self.height
            self.height = self.FIXED_IMAGE_HEIGHT
            self.width = int(resize_ratio*self.width)
            self.sudoku_img = cv2.resize(self.sudoku_img, (self.width, self.height))

        #Applying Filter and doing some preprocessing

        image = cv2.bilateralFilter(self.sudoku_img, 9, 15, 15)
        image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3)
        image = cv2.bitwise_not(image)
        (cnts, _) = cv2.findContours(image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
        grid = None
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            if len(approx) == 4:
                grid = approx
                break
        pt1 = np.float32([grid[0][0], grid[1][0], grid[2][0], grid[3][0]])
        #print pt1
        pt2 = np.float32([[0, 0], [0, self.height], [self.width, self.height], [self.width, 0]])
        M = cv2.getPerspectiveTransform(pt1, pt2)
        return cv2.warpPerspective(image, M, (self.width, self.height))

    def _extract_digit(self, x, y):
        print "{0},{1}".format(x, y)
        block = self.sudoku_img[(y)*self.height/9:(y+1)*self.height/9, x*self.width/9:(x+1)*self.width/9]
        return cv2.resize(block, (28, 28))

    def _has_digit(self, img_block):
        Sudoku.display(img_block)
        img_block = img_block[10:18, 10:18]
        average_row = np.average(img_block, axis=0)
        average = np.average(average_row, axis=0)
        print average
        return average

    def find_digits(self):
        avg = 0
        for x in range(9):
            for y in range(9):
                block = self._extract_digit(x, y)

                avg += self._has_digit(block)

        print "Image average", avg/81


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



im = open('../data/sudoku_3.png',
          'r')
sudoku = Sudoku(im)
sudoku.find_digits()
#b = sudoku._extract_digit(1,3)
#sudoku._has_digit(b)

