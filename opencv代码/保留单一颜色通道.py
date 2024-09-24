import numpy as np
import cv2
import matplotlib.pyplot as plt

# 读取1.jpg图片的彩色图像
img = cv2.imread('1.jpg' , cv2.IMREAD_COLOR) # 读取彩色图像(RGB图像)

cur_img  = img.copy()# 复制一份图片（将img复制给cur_img）

# 以下操作前面两个“：”是指取整个图片（之前是写值来cut图片），后面的数字代表通道（0代表blue通道，1代表green通道，2代表red通道）
# 经过以下操作后，图片的blue通道和green通道的值全部变为0，只剩下red通道的值，图片便只有红色这个颜色了
cur_img[:,:,0] = 0 # 将blue通道的值全部变为0
cur_img[:,:,1] = 0 # 将green通道的值全部变为0

cv2.imshow('good' , cur_img)
cv2.waitKey(0)
cv2.destroyAllWindows()