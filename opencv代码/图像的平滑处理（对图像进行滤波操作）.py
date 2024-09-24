import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('7.png' , cv2.IMREAD_COLOR) # 读取彩色图像(RGB图像)

# 均值滤波法
# 均值滤波就是用卷积核中所有值的均值去代替当前像素点的值，这样就可以达到去噪的目的
# 这种做法能够减弱噪声的强度，但是会使图像变得模糊
blur = cv2.blur(img , (5 , 5)) # (3 ,  3)表示卷积核的大小

res = np.hstack((img , blur))

cv2.namedWindow('blur', cv2.WINDOW_NORMAL) # 设置窗口标题
cv2.resizeWindow('blur', 800, 400)  # 你可以根据需要设置不同的宽度和高度

cv2.imshow('blur' , res)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 高斯滤波函
# 高斯模糊的卷积核里的数值是满足高斯分布的，相当于其更重视中间的数值,越靠边上的函数占的权重越小
# 函数原型：dst = cv2.GaussianBlur(src , ksize , sigmaX , dst , sigmaY , borderType)
# ——src输入图像。
# ——ksize高斯内核大小。 ksize.width和ksize.height可以不同，但​​它们都必须为正数和奇数，也可以为零，然后根据sigmaX和sigmaY计算得出。
# ——dst输出图像的大小和类型与src相同。
# ——sigmaX X方向上的高斯核标准偏差。
# ——sigmaY Y方向上的高斯核标准差；如果sigmaY为零，则将其设置为等于sigmaX；如果两个sigmas为零，则分别从ksize.width和ksize.height计算得出；为了完全控制结果，而不管将来可能对所有这些语义进行的修改，建议指定所有ksize，sigmaX和sigmaY。
aussian = cv2.GaussianBlur(img , (5 , 5) , 0) # (5 , 5)表示卷积核的大小,0表示标准差取0

res = np.hstack((img , aussian))

cv2.namedWindow('GaussianBlur', cv2.WINDOW_NORMAL) # 设置窗口标题
cv2.resizeWindow('GaussianBlur', 800, 400)  # 你可以根据需要设置不同的宽度和高度

cv2.imshow("GaussianBlur" , res)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 中值滤波法（将内核中的值大小排序取中间的值作为中间点的值）
# 函数原型：dst = cv2.medianBlur(src , ksize , dst)
# ——src输入1、3或8位图像。当ksize=3或5时，图像深度应为CV_8U，CV_16U或CV_32F，当ksize=7时，图像深度仅限于CV_8U。
# ——dst与src大小和类型相同。
# ——ksize是大于1的奇数。
median = cv2.medianBlur(img , 5) # 5表示卷积核的大小

cv2.namedWindow('medianBlur', cv2.WINDOW_NORMAL) # 设置窗口标题
cv2.resizeWindow('medianBlur', 800, 400)  # 你可以根据需要设置不同的宽度和高度

res = np.hstack((img , median))

cv2.imshow("medianBlur" , res)
cv2.waitKey(0)
cv2.destroyAllWindows()

#***************************************总结*****************************************#
# 中值滤波法：适合去除噪声点（因为噪声点的值非常大而取中值的时候就不可能取到噪声点所以有利于去除噪声点）
# 均值滤波法：适合去除高斯噪声（因为高斯噪声是均匀分布的所以用均值滤波法去除）
# 高斯滤波法：适合去除高斯噪声（因为高斯滤波法的卷积核是满足高斯分布的所以用高斯滤波法去除）
