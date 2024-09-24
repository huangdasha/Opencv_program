import numpy as np
import cv2
import matplotlib.pyplot as plt

def cv_show(name , img):
    cv2.imshow(name , img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 其中sobel算子中重要的就是两个量Gx和Gy，这两个量的计算公式如图片所示
# Gx是相当于水平的梯度，Gy是相当于垂直的梯度，Gx是看当前像素点的左右两个点的差值，Gy是看当前像素点的上下两个点的差值
# img = cv2.imread('example1.png' , cv2.IMREAD_UNCHANGED) # 读取原图
img2 = cv2.imread('5.png' , cv2.IMREAD_GRAYSCALE) # 读取灰度图像
# cv_show('Principle' , img)

# ******************************************计算sobel算子****************************************************#
# sobel算子计算梯度函数(dst = cv2.Sobel(src , ddepth , dx , dy , ksize = x))
# 其中ddepth表示图像的深度，dx和dy表示求导的阶数，ksize表示卷积核的大小
sobelx = cv2.Sobel(img2 , cv2.CV_64F , 1 , 0 , ksize = 3) # x方向的梯度(这里的64F表示64位浮点数，可以取负值，因为差值可能为负值)
sobely = cv2.Sobel(img2 , cv2.CV_64F , 0 , 1 , ksize = 3) # y方向的梯度

# 但是如果取负值的话当右边减左边的时候右边的边界就无法体现，上下也是一样的，所以我们要取绝对值
sobelx2 = cv2.convertScaleAbs(sobelx) # x方向的梯度(取绝对值)
sobely2 = cv2.convertScaleAbs(sobely) # y方向的梯度(取绝对值)

cv2.imshow('original picture' , img2)

changed = np.hstack((sobelx , sobely , sobelx2 , sobely2)) # 将原图像，腐蚀操作后的图像，膨胀操作后的图像横向拼接起来
cv2.imshow("Changed picture" , changed)

sobelxy = cv2.addWeighted(sobelx2 , 0.5 , sobely2 , 0.5 , 0) # 将x方向和y方向的梯度加起来
cv2.imshow('Final picture' , sobelxy)

cv2.waitKey(0)
cv2.destroyAllWindows()