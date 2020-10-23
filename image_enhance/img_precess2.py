# -*- coding: UTF-8 -*-
import glob
import os.path

import cv2
from PIL import Image
from PIL import ImageEnhance


def img_enhance(img_path, new_name_enhandce, new_path_enhance):
    # 原始图像
    image = Image.open(img_path)
    # image.show()

    # 亮度增强
    light_name = 'light_' + new_name_enhance
    light_path = os.path.join(new_path_enhance, light_name)
    enh_bri = ImageEnhance.Brightness(image)
    brightness = 1.3
    image_brightened = enh_bri.enhance(brightness)
    image_brightened.save(light_path)
    # image_brightened.show()

    # 色度增强
    chroma_name = 'chroma_' + new_name_enhance
    chroma_path = os.path.join(new_path_enhance, chroma_name)
    enh_col = ImageEnhance.Color(image)
    color = 1.5
    image_colored = enh_col.enhance(color)
    image_colored.save(chroma_path)
    # image_colored.show()
    #
    # 对比度增强
    contrast_name = 'contrast_' + new_name_enhance
    contrast_path = os.path.join(new_path_enhance, contrast_name)
    enh_con = ImageEnhance.Contrast(image)
    contrast = 1.5
    image_contrasted = enh_con.enhance(contrast)
    image_contrasted.save(contrast_path)
    # image_contrasted.show()
    #
    # 锐度增强
    acutance_name = 'acutance_' + new_name_enhance
    acutance_path = os.path.join(new_path_enhance, acutance_name)
    enh_sha = ImageEnhance.Sharpness(image)
    sharpness = 3.0
    image_sharped = enh_sha.enhance(sharpness)
    image_sharped.save(acutance_path)
    # image_sharped.show()


def img_flip_rotation(img_path, new_path1, new_path2, new_path3):
    # read image
    img = cv2.imread(img_path, cv2.IMREAD_ANYCOLOR)
    # cv2.imshow('img_raw', img)
    # cv2.waitKey()
    height, width, temp = img.shape
    # do flip
    xImg = cv2.flip(img, 1, dst=None)
    cv2.imwrite(new_path1, xImg)
    # do rotation
    M = cv2.getRotationMatrix2D((width / 2, 0), 10, 1)
    img_ro = cv2.warpAffine(img, M, img.shape[:2])
    cv2.imwrite(new_path2, img_ro)
    # do rotation's mirror
    M2 = cv2.getRotationMatrix2D((width / 2, 0), 350, 1)
    img_ro_mirror = cv2.warpAffine(img, M2, img.shape[:2])
    cv2.imwrite(new_path3, img_ro_mirror)


if __name__ == '__main__':
    postfix = '.jpg'
    file_root = '..\\dataprocess_dir\\'
    for filename in os.listdir(r"..\\dataprocess_dir"):  # listdir的参数是文件夹的路径
        fold_path = []
        fold_path.extend(glob.glob(os.path.join(file_root, filename, '*{}'.format(postfix))))
        # print(len(fold_path))
        for i in range(len(fold_path)):
            # for fold in fold_path:
            fold = fold_path[i]
            print(fold)
            img_path = fold
            new_name1 = 'flip_' + fold.split('\\')[-1]
            new_name2 = 'rota_' + fold.split('\\')[-1]
            new_name3 = 'rota_mirror' + fold.split('\\')[-1]
            new_path1 = os.path.join(file_root, filename, new_name1)
            new_path2 = os.path.join(file_root, filename, new_name2)
            new_path3 = os.path.join(file_root, filename, new_name3)

            # do image process
            img_flip_rotation(img_path, new_path1, new_path2, new_path3)

    # do image enhance
    postfix = '.jpg'
    file_root = '..\\dataprocess_dir\\'
    for filename in os.listdir(r"..\\dataprocess_dir"):  # listdir的参数是文件夹的路径
        fold_path = []
        fold_path.extend(glob.glob(os.path.join(file_root, filename, '*{}'.format(postfix))))
        # print(len(fold_path))
        for i in range(len(fold_path)):
            # for fold in fold_path:
            fold = fold_path[i]
            print(fold)
            img_path = fold
            new_path_enhance = os.path.join(file_root, filename)
            new_name_enhance = fold.split('\\')[-1]
            img_enhance(img_path, new_name_enhance, new_path_enhance)
