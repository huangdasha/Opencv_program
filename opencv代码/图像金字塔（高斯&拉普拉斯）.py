import numpy as np
import cv2
import matplotlib.pyplot as plt

# 高斯金字塔简单来说就是通过先卷积后缩小和先放大再卷积两种方式来进行图像的缩放
# 读取图像
img = cv2.imread('2.jpg')
cv2.imshow('img' , img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 函数原型：cv2.pyrDown(image, dst=None, dstsize=None, borderType=None)
# 对图像先进行卷积再将所有的偶数行全部删掉，这样就得到了原来图像二分之一大小的图像
img2 = cv2.pyrDown(img)
cv2.imshow('img2' , img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 函数原型：cv2.pyrUp(image, dst=None, dstsize=None, borderType=None)
# 对图像先进行放大(比如2✖2 -> 4✖4，加的所有行值都为0)，然后再进行卷积，这样就得到了原来图像的两倍大小的图像
img3 = cv2.pyrUp(img)
cv2.imshow('img3' , img3)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 拉普拉斯金字塔是通过高斯金字塔得到的图像进行还原得到的图像
# 简要公式：L(i) = G(i) - pyrUp(pyrDown(G(i+1)))(原图像 - 先缩小再放大的图像)
# 函数原型：cv2.subtract(src1, src2, dst=None, mask=None, dtype=None)
img4 = cv2.pyrUp(img2)# 将之前缩小的图像放大
img5 = cv2.subtract(img , img4)# 这里是原图像减去先缩小再放大的图像(函数是减法函数)
cv2.imshow('img5' , img5)
cv2.waitKey(0)
cv2.destroyAllWindows()