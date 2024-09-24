import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('4.png' , cv2.IMREAD_UNCHANGED) # 读取灰度图像
kernel = np.ones((20 , 20) , np.uint8) # 定义一个5 * 5的卷积核，相当于卷积核内都为白色这个核就为白色否则就是黑色

# 开闭运算就是膨胀和腐蚀操作的先后顺序，通过cv2.morphologyEx()函数中cv2.MORPH_OPEN参数来实现
# 梯度运算：膨胀 - 腐蚀，得到轮廓图像，通过cv2.morphologyEx()函数中cv2.MORPH_GRADIENT参数来实现
# 开运算：先腐蚀再膨胀(消除毛刺)
opening = cv2.morphologyEx(img , cv2.MORPH_OPEN , kernel) # 开运算，即先腐蚀后膨胀
 
# 闭运算：先膨胀再腐蚀（放大毛刺）
closing = cv2.morphologyEx(img , cv2.MORPH_CLOSE , kernel) # 闭运算，即先膨胀后腐蚀

# 梯度运算
gradient = cv2.morphologyEx(img , cv2.MORPH_GRADIENT , kernel) # 梯度运算，即膨胀图像减去腐蚀图像

cv2.namedWindow('result', cv2.WINDOW_NORMAL) # 设置窗口标题
cv2.resizeWindow('result', 1600, 600)  # 你可以根据需要设置不同的宽度和高度

cv2.imshow("oraginal" , img)
cv2.waitKey(0)
cv2.destroyAllWindows()

res = np.hstack((opening , closing , gradient)) # 将原图像，腐蚀操作后的图像，膨胀操作后的图像横向拼接起来

# 下面两张图是原图->开运算图->闭运算图,以及膨胀迭代次数不同的图，可以明显看到膨胀操作的作用
cv2.imshow("result", res)# 膨胀操作后的图像
cv2.waitKey(0)
cv2.destroyAllWindows()
