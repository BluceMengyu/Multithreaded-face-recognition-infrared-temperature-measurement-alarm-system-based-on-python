import cv2
from PIL import Image
from PIL import ImageEnhance
img = cv2.imread('1.jpg')
blur = cv2.bilateralFilter(img,9,75,75)

alpha = 0.3
beta = 1-alpha
gamma = 0
img_add = cv2.addWeighted(img, alpha, blur, beta, gamma)
cv2.imwrite('img_add.jpg', img_add)
# # 锐度增强
img_add = Image.open('img_add.jpg')
#img_add.show()
enh_sha = ImageEnhance.Sharpness(img_add)
sharpness = 1.5
image_sharped = enh_sha.enhance(sharpness)
#image_sharped.show()

# # 对比度增强
enh_con = ImageEnhance.Contrast(image_sharped)
contrast = 1.15
image_contrasted = enh_con.enhance(contrast)
image_contrasted.show()

cv2.waitKey(0)