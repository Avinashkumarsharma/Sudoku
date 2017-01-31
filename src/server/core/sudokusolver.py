#!/usr/bin/env python
"""
Author: Avik
Date: 30th Dec 2016
Description : Solver for sudoku from image

"""

import cv2
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras import backend as K
K.set_image_dim_ordering('th')

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

    @staticmethod
    def printSudoku(mat):
        for i in range(9):
            print mat[i*9:i*9+9]


    def __init__(self, image):
        image_byte = image.read()
        self.sudoku_img = cv2.imdecode(np.fromstring(image_byte, np.uint8), cv2.IMREAD_GRAYSCALE)
        self.height, self.width = self.sudoku_img.shape
        self.sudoku_img = self._preprocess()
        self.download_data()
        self.features = np.array(self.dataset.data, 'int16')
        self.label = np.array(self.dataset.target, 'int')

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
        (cnts, _) = cv2.findContours(image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
        grid = None
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            if len(approx) == 4:
                grid = approx
                break
        pt1 = np.float32([grid[0][0], grid[1][0], grid[2][0], grid[3][0]])
        pt2 = np.float32([[0, 0], [0, self.height], [self.width, self.height], [self.width, 0]])
        M = cv2.getPerspectiveTransform(pt1, pt2)
        _img = cv2.warpPerspective(image, M, (self.width, self.height))
        return _img

    def _extract_digit(self, x, y, erode = True):
        #print "{0},{1}".format(x, y)
        block = self.sudoku_img[(y)*self.height/9:(y+1)*self.height/9, x*self.width/9:(x+1)*self.width/9]
        _img = cv2.resize(block, (38, 38))
        if erode:
            return cv2.erode(_img, np.ones((3,3)))
        else:
            return _img


    def _post_process(self, img):
        img = cv2.resize(img, (28, 28))
        # centering the image by finding bounding box
        cont, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        area = [cv2.contourArea(c) for c in cont]
        max_index = np.argmax(area)
        cnt = cont[max_index]
        x, y, w, h = cv2.boundingRect(cnt)
        digit_img = img[y:y + h, x:x + w]
        width = int(w * (20 / float(h)))
        height = 20
        im = cv2.resize(digit_img, (width, height))
        _x = (28 - width) / 2
        final_img = np.zeros(shape=(28, 28))
        final_img[4:24, _x:_x + width] = im
        return final_img


    def _has_digit(self, img_block):
        (cnts, _) = cv2.findContours(img_block.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        if len(cnts) > 0:
            return cv2.contourArea(cnts[0])
        return 0

    def _find_digits_in_row(self, y):
        digit = []
        for x in range(9):
            img_block = self._extract_digit(x, y)[5:33, 5:33]
            digit.append(self._has_digit(img_block[6:20, 6:20]))
        return digit

    def decode_sudoku(self):
        sudoku = []
        predictor = Predictor()
        for y in range(9):
            print y
            row = self._find_digits_in_row(y)
            _row = []
            for x, val in enumerate(row):
                if val > 0:
                    img_block = self._extract_digit(x, y)
                    #Sudoku.display(img_block)
                    try:
                        _img = self._post_process(img_block)
                    except ValueError:
                        print "Value Eror Ho gaya!!"
                        _img = self._post_process(self._extract_digit(x, y, erode=False))
                    _row.append(predictor.predict(_img))
                else:
                    _row.append(-1)
            sudoku.append(_row)
        return sudoku

class Predictor:

    SEED = 7

    def __init__(self):
        np.random.seed(Predictor.SEED)

    def train(self):
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        # reshape to be [samples][pixels][width][height]
        X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
        X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')
        X_train = X_train / 255
        X_test = X_test / 255
        # one hot encode outputs
        y_train = np_utils.to_categorical(y_train)
        y_test = np_utils.to_categorical(y_test)
        num_classes = y_test.shape[1]
        model = self.baseline_model(num_classes)
        # Fit the model
        model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=10, batch_size=200, verbose=2)
        # Final evaluation of the model
        model.save_weights("digit_weights.h5")
        scores = model.evaluate(X_test, y_test, verbose=0)
        print("Baseline Error: %.2f%%" % (100 - scores[1] * 100))

    def baseline_model(self, num_classes=10):
        # create model
        model = Sequential()
        model.add(Convolution2D(32, 5, 5, border_mode='valid', input_shape=(1, 28, 28), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def predict(self, img, weights='digit_weights.h5'):
        img_flatten = img.reshape(1, 1, 28, 28).astype('float32')/255
        model = self.baseline_model()
        model.load_weights(weights)
        return model.predict_classes(img_flatten, verbose=1)[0]





im = open('../data/sudoku_4.png', 'rb')
sudoku = Sudoku(im)

#print sudoku.decode_sudoku()
'''img = sudoku._extract_digit(3,0)
img = cv2.dilate(img, np.ones((2,2)))
Sudoku.display(img)
try:
    _img = sudoku._post_process(img)
except ValueError:
    print "Value error ho gaya hai !!'"
    _img = sudoku._post_process(sudoku._extract_digit(6, 0, erode=False))

p = Predictor()
print p.predict(_img)
'''
