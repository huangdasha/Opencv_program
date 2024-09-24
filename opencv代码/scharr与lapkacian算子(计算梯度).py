import numpy as np
import cv2
import matplotlib.pyplot as plt

# **************************两种算子原理************************** #
# scharr算子和sobel算子基本是一样的只不过矩阵中取的值更大这样就让差值更大，从而让边缘更加明显
# Laplacian算子是二阶导数，所以对噪声比较敏感，所以一般不用来检测边缘，但是可以用来检测图像中的孤立点或者是噪声点

# **************************scharr算子计算原理************************** #
# 下面是scharr算子的原理矩阵
principle = cv2.imread('principle of scharr.png' , cv2.IMREAD_GRAYSCALE) # 读取灰度图像
cv2.imshow('Principle' , principle)


# 下面是几种算子的对比
img = cv2.imread('6.png' , cv2.IMREAD_GRAYSCALE) # 读取灰度图像

# sobel算子函数(dst = cv2.Sobel(src , ddepth , dx , dy , ksize = x))
sobelx = cv2.Sobel(img , cv2.CV_64F , 1 , 0 , ksize = 3) # x方向的梯度(这里的64F表示64位浮点数，可以取负值，因为差值可能为负值)
sobely = cv2.Sobel(img , cv2.CV_64F , 0 , 1 , ksize = 3) # y方向的梯度
sobelx2 = cv2.convertScaleAbs(sobelx) # x方向的梯度(取绝对值)
sobely2 = cv2.convertScaleAbs(sobely) # y方向的梯度(取绝对值)
sobelxy = cv2.addWeighted(sobelx2 , 0.5 , sobely2 , 0.5 , 0) # 将x方向和y方向的梯度加起来

# scharr算子函数(dst = cv2.Scharr(src , ddepth , dx , dy))
scharrx = cv2.Scharr(img , cv2.CV_64F , 1 , 0)
scharry = cv2.Scharr(img , cv2.CV_64F , 0 , 1)
scharrx2 = cv2.convertScaleAbs(scharrx) # x方向的梯度(取绝对值)
scharry2 = cv2.convertScaleAbs(scharry) # y方向的梯度(取绝对值)
scharrxy = cv2.addWeighted(scharrx2 , 0.5 , scharry2 , 0.5 , 0) # 将x方向和y方向的梯度加起来

# laplacian算子函数(dst = cv2.Laplacian(src , ddepth))
laplacian = cv2.Laplacian(img,cv2.CV_64F)
laplacian2 = cv2.convertScaleAbs(laplacian) # x方向的梯度(取绝对值)

res = np.hstack((img , sobelxy , scharrxy , laplacian2))  # 将原图像和几种算法得到的图像横向拼接起来

cv2.imshow("res" , res)
cv2.waitKey(0)
cv2.destroyAllWindows()
# 看结果就知道scharr算子算出来的更加细腻，边缘更加明显，但是也会带来更多的噪声
# laplacian算子的结果是黑白分明，但是噪声也很多，所以一般不用来检测边缘，但是可以用来检测图像中的孤立点或者是噪声点