import numpy as np
import cv2

# canny边缘检测的步骤
# 1): 使用高斯滤波，平滑图片、消除噪声
# 2): 计算图像中每个像素点的梯度强度（大小）和方向
# 3): 应用非极大值抑制，以消除边缘检测带来的杂散响应（在多个框中选择较准确的框）
#     非极大值抑制宏观原理：https://blog.csdn.net/shuzfan/article/details/52711706
# 4): 应用双阈值检测来确定真实的和潜在的边缘
# 5): 通过抑制孤立的弱边缘最终完成边缘检测
img = cv2.imread('6.png' , cv2.IMREAD_GRAYSCALE) # 读取灰度图像

v1 = cv2.Canny(img , 80 , 150) # 80是低阈值，150是高阈值
v2 = cv2.Canny(img , 50 , 100) # 50是低阈值，100是高阈值

res = np.hstack((img , v1 , v2))  # 将原图像和几种算法得到的图像横向拼接起来

cv2.imshow("res" , res)
cv2.waitKey(0)
cv2.destroyAllWindows()