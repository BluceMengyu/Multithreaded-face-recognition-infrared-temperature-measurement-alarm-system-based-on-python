# -*- coding: UTF-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt

def main():
    img = cv2.imread("4.jpg",0)
    eq = cv2.equalizeHist(img)
    cv2.imshow("Histogram Equalization", np.hstack([img, eq]))
    cv2.waitKey(0)


if __name__ == '__main__':
    main()



