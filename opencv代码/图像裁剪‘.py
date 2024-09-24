import numpy as np
import cv2
import matplotlib.pyplot as plt

# 读取1.jpg图片的彩色图像
img = cv2.imread('1.jpg' , cv2.IMREAD_COLOR) # 读取彩色图像(RGB图像)

# 截取图像的一部分(第一个参数是wideth ， 第二个参数是hight)
picture = img[0:200 , 0:800]# 表示的意思是截取图像的左上角的200*800的部分（就是宽度0~200，高度0~800部分）

# 这里是将一个图片的构成分离成三个变量了（因为图片是由）
b , g , r = cv2.split(picture) # 将图像的三个通道分离出来(blue grenn red)并且赋值给三个参数（注意！！一定要搞清楚顺序）

print("蓝色颜色的矩阵为：" , b)# 打印出blue颜色通道的矩阵
print("蓝色颜色的矩阵大小为：" , b.shape)# 打印出blue颜色通道的矩阵的大小

cv2.imshow('good' , picture)
cv2.waitKey(0)
cv2.destroyAllWindows()