#!/usr/bin/env python
from flask import Flask, request, redirect, url_for
import sys
import cv2
import numpy as np
app = Flask(__name__)


def display(image, name='Image', duration=0, resize=True):
    print 'Showing image'
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, image)
    cv2.waitKey(duration)
    cv2.destroyAllWindows()

@app.route('/')
def hello_world():
    return 'Hello world!'

@app.route('/process', methods=['GET', 'POST'])
def process_image():
    if request.method == 'GET':
        return 'USE POST'
    sys.stderr.write(str(request.files))
    if 'file' in request.files:
        _file = request.files['file'].read()
        im = cv2.imdecode(np.fromstring(_file, np.uint8), cv2.CV_LOAD_IMAGE_GRAYSCALE)
        sys.stderr.write(str(im.shape))
        #display(im)
        #sys.stderr.write(_file)
        return "Bye Bye !!"
    return 'Deafult\n'

def try_open_file():
    f = open('sudoku.jpg', 'rw')
    image = np.frombuffer(f.read(), dtype='uint8')
    im = cv2.imdecode(image, 1)
    print im.shape
    #display(im)
#try_open_file()





