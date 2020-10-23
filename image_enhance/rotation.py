# -*- coding: UTF-8 -*-
import cv2
import numpy as np

def main():

   img = cv2.imread('3.jpg')
   height, width, temp = img.shape
   M = cv2.getRotationMatrix2D((width/2, height/2), 30, 1)
   img_ro = cv2.warpAffine(img, M, img.shape[:2])
   cv2.imshow('img_ro', img_ro)
   cv2.imshow('image', img)
   cv2.waitKey(0)

if __name__ == '__main__':
    main()



