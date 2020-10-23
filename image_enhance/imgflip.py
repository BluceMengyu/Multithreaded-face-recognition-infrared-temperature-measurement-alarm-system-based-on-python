# -*- coding: UTF-8 -*-
import cv2
import numpy as np

def main():

   img = cv2.imread('3.jpg')
   height, width, temp = img.shape
   xImg = cv2.flip(img, 1, dst=None)
   yImg = cv2.flip(img, 0, dst=None)
   cv2.imshow('xImg', xImg)
   #cv2.imshow('yImg', yImg)
   cv2.imshow('image', img)
   cv2.waitKey(0)

if __name__ == '__main__':
    main()



