# -*- coding: UTF-8 -*-
import cv2
import numpy as np

def main():

   img = cv2.imread('2.jpg')
   height, width, temp = img.shape
   downscale = cv2.resize(img, (100, 100), interpolation=cv2.INTER_LINEAR)
   upscale = cv2.resize(img,(2*width, 2*height), interpolation=cv2.INTER_LINEAR)
   cv2.imshow('downscale', downscale)
   cv2.imshow('upscale', upscale)
   cv2.imshow('image', img)
   cv2.waitKey(0)

if __name__ == '__main__':
    main()



