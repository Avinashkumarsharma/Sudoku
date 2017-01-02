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
    def __init__(self, image):
        self.sudoku_img = cv2.imread(image)
        self.height, self.width = self.sudoku_img.shape[0], self.sudoku_img.shape[1]

    def _preprocess(self):
        if self.height > 500:
            pass


sudoku = Sudoku('../data/sudoku.jpg')
