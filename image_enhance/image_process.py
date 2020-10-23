# -*- coding: UTF-8 -*-
from PIL import Image
from PIL import ImageEnhance
import cv2
import os
import os.path
import glob
import cv2
import numpy as np
def img_read(img_path):
    img = cv2.imread(img_path,cv2.IMREAD_ANYCOLOR)
    cv2.imshow('img_raw',img)
    return img

def imgflip(img_path,res_path):
    img = cv2.imread(img_path,cv2.IMREAD_ANYCOLOR)
    height, width, temp = img.shape
    xImg = cv2.flip(img, 1, dst=None)
    cv2.imwrite(res_path, xImg)
    return xImg

def rotation(img_path, res_path):
    img = cv2.imread(img_path,cv2.IMREAD_ANYCOLOR)
    height, width, temp = img.shape
    M = cv2.getRotationMatrix2D((width/2, 0), 20, 1)
    img_ro = cv2.warpAffine(img, M, img.shape[:2])
    cv2.imwrite(res_path, img_ro)
    #cv2.imshow('img_ro', img_ro)
    return img_ro


def img_enhance(img_path):
    # 原始图像
    image = Image.open(img_path)
    #image.show()

    # 亮度增强
    enh_bri = ImageEnhance.Brightness(image)
    brightness = 1.3
    image_brightened = enh_bri.enhance(brightness)
    image_brightened.show()

    # # 色度增强
    # enh_col = ImageEnhance.Color(image)
    # color = 1.5
    # image_colored = enh_col.enhance(color)
    # image_colored.show()
    #
    # # 对比度增强
    # enh_con = ImageEnhance.Contrast(image)
    # contrast = 1.5
    # image_contrasted = enh_con.enhance(contrast)
    # image_contrasted.show()
    #
    # # 锐度增强
    # enh_sha = ImageEnhance.Sharpness(image)
    # sharpness = 3.0
    # image_sharped = enh_sha.enhance(sharpness)
    # image_sharped.show()

def convertjpg(jpgfile, outdir, width=128, height=128):

    src = cv2.imread(jpgfile, cv2.IMREAD_ANYCOLOR)
    try:
        dst = cv2.resize(src, (width, height), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(outdir, os.path.basename(jpgfile)), dst)
    except Exception as e:
        print(e)

if __name__ == '__main__':
    postfix = '.jpg'
    file_root = '..\\dataprocess_dir\\'
    for filename in os.listdir(r"..\\dataprocess_dir"):  # listdir的参数是文件夹的路径
        fold_path = []
        fold_path.extend(glob.glob(os.path.join(file_root, filename, '*{}'.format(postfix))))
        #print(len(fold_path))
        for i in range(len(fold_path)):
        #for fold in fold_path:
            fold = fold_path[i]
            print(fold)
            img_path = fold
            new_name1 = 'flip_'+fold.split('\\')[-1]
            new_name2 = 'rota_' + fold.split('\\')[-1]
            new_path1 = os.path.join(file_root, filename, new_name1)
            new_path2 = os.path.join(file_root, filename, new_name2)
            print(new_path1,new_path2)
            img_raw = img_read(img_path)
            # cv2.imshow('img_raw', img_raw)
            # cv2.waitKey()
            imgflip = imgflip(img_path, new_path1)
            # cv2.imshow('img_flip', imgflip)
            # cv2.waitKey()
            #cv2.imwrite(new_path1, imgflip)
            imgro = rotation(img_path, new_path2)
            # cv2.imshow('img_ro', imgro)
            # #cv2.imwrite(new_path2, imgro)
            # cv2.waitKey()

    # img_path = '3.jpg'
    # img_raw = img_read(img_path)
    # imgflip = imgflip(img_raw)
    # imgro = rotation(imgflip)
    # #img_enhance(img_path)
    # cv2.imshow('img_raw',img_raw)
    # cv2.imshow('xImg', imgflip)
    # cv2.imshow('img_ro', imgro)
    # cv2.waitKey(0)

