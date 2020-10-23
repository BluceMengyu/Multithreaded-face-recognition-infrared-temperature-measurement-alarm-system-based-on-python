import cv2
import numpy as np
from PIL import Image
from PIL import ImageEnhance


def beautiful(img):
    # 滤波
    blur = cv2.bilateralFilter(img, 14, 40, 40)
    # 融合
    alpha = 0.3
    beta = 1 - alpha
    gamma = 0
    img_add = cv2.addWeighted(img, alpha, blur, beta, gamma)
    img_add = Image.fromarray(cv2.cvtColor(img_add, cv2.COLOR_BGR2RGB))
    enh_sha = ImageEnhance.Sharpness(img_add)
    # 增强
    sharpness = 1.5
    image_sharped = enh_sha.enhance(sharpness)
    # 锐化
    enh_con = ImageEnhance.Contrast(image_sharped)
    contrast = 1.15
    image_contrasted = enh_con.enhance(contrast)
    image_contrasted = cv2.cvtColor(np.asarray(image_contrasted), cv2.COLOR_RGB2BGR)
    return image_contrasted


if __name__ == '__main__':
    img = cv2.imread("4.jpg",1)
    img_b = beautiful(img)
    cv2.imshow('beautiful',img_b)
    cv2.waitKey()

