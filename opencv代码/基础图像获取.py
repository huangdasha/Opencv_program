import numpy as np
import cv2
import matplotlib.pyplot as plt

# 宏定义一个函数用来显示图片的
def img_show(title , img):
    cv2.namedWindow(title, cv2.WINDOW_NORMAL) # 设置窗口标题
    cv2.resizeWindow(title, 800, 600)  # 你可以根据需要设置不同的宽度和高度
    cv2.imshow(title , img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
# 读取1.jpg图片的彩色图像
img = cv2.imread('1.jpg' , cv2.IMREAD_COLOR) # 读取彩色图像(RGB图像)
# 读取1.jpg图片的灰度图像
img = cv2.imread('1.jpg' , cv2.IMREAD_GRAYSCALE)# 读取灰度图像（灰度图像）

# 设置窗口大小
cv2.namedWindow('result', cv2.WINDOW_NORMAL) # 设置窗口标题
cv2.resizeWindow('result', 800, 600)  # 你可以根据需要设置不同的宽度和高度

# 图像的一些属性
print("图像尺寸为" , img.shape) # 打印图像的尺寸
print("图像数据类型为" , img.dtype) # 打印图像的数据类型
print("图像大小为" , img.size) # 打印图像的大小

cv2.imshow('result' , img)
#等待时间 ，0表示任意按键按下就终止（其实就是一直等待，知道按下一个按键之后就会之下下面的销毁所有窗口函数）
cv2.waitKey(0)
#如若设置为5000则表示在5s后执行下面的销毁函数
#cv2.waitKey(5000)
#销毁所有窗口（这样就关掉了显示照片的窗口） 
cv2.destroyAllWindows()

img_show('result' , img)