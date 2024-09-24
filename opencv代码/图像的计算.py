import numpy as np
import cv2
import matplotlib.pyplot as plt

def img_show(img , title):
    cv2.namedWindow(title, cv2.WINDOW_NORMAL) # 设置窗口标题
    cv2.imshow(title , img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
img = cv2.imread('1.jpg' , cv2.IMREAD_COLOR) # 读取彩色图像(RGB图像)


# ******************************以下是数值加法问题*********************************** #
# 数值的加法问题（直接加）
img2 = img + 10 # 图像的每个像素点都加10

# 计算结果比如249 + 10 = 259，但是uint8类型的图像最大值只能是255，所以会溢出，溢出后的结果就是259 % 256 = 3
print('img is :' , img[:5 , :])
print('img2 is :' , img2[:5 , :])

# 数值加法问题（函数加）
img2 = cv2.add(img , 10) # 图像的每个像素点都加10（会将结果限制在范围中）

# 计算结果比如249 + 10 = 259，但是uint8类型的图像最大值只能是255，所以会溢出，但这个函数会自动将溢出的值截断为255即超出取最大
print('img is :' , img[:5 , :])
print('img2 is :' , img2[:5 , :])


# ******************************以下是通过公式来改变数值问题*********************************** #

# 1.倍数放大(或缩小)公式
img = cv2.imread('1.jpg' , cv2.IMREAD_COLOR) # 读取彩色图像(RGB图像)

img_3x3y = cv2.resize(img , (0 , 0) , fx = 3 , fy = 3) # 这就是将图片的wideth and height放大三倍
img_2x2y = cv2.resize(img , (0 , 0) , fx = 2 , fy = 2) # 这就是将图片的wideth and height放大二倍
img_3x1y = cv2.resize(img , (0 , 0) , fx = 3 , fy = 1) # 这就是将图片的wideth放大三倍
img_1x3y = cv2.resize(img , (0 , 0) , fx = 1 , fy = 3) # 这就是将图片的height放大三倍


# ******************************以下是图像融合问题*********************************** #
# 当对于两个图像大小不一样的图片想进行融合（即相加时）不行，这时便需要将两个图像的大小统一

# 图像融合low（直接函数加方法（前提是两张图片大小相等））
# 函数原型：cv2.add(src1, src2, dst=None, mask=None, dtype=None)
img = cv2.imread('1.jpg' , cv2.IMREAD_COLOR) # 读取彩色图像(RGB图像)
img2 = cv2.imread('2.jpg' , cv2.IMREAD_COLOR) # 读取彩色图像(RGB图像)

img = cv2.resize(img , (512 , 512)) # 将img的大小裁剪为为512 * 512
img2 = cv2.resize(img2 , (512 , 512)) # 将img2的大小裁剪为为512 * 512

img_all = cv2.add(img , img2) # 将两个图像的像素点相加（会将结果限制在范围中）


# 图像融合high（带有公式计算的函数）
# 函数原型：cv2.addWeighted(src1, alpha, src2, beta, gamma, dst=None, dtype=None)
# 函数公式：dst = src1*alpha + src2*beta + gamma（所以alpha和beta相当于两张图象的占比，再加一个gamma提高整体的亮度）
img = cv2.imread('1.jpg' , cv2.IMREAD_COLOR) # 读取彩色图像(RGB图像)
img2 = cv2.imread('2.jpg' , cv2.IMREAD_COLOR) # 读取彩色图像(RGB图像)

img = cv2.resize(img , (512 , 512)) # 将img的大小裁剪为为512 * 512
img2 = cv2.resize(img2 , (512 , 512)) # 将img2的大小裁剪为为512 * 512

img_all1 = cv2.addWeighted(img , 0.3 , img2 , 0.7 , 0)# 相当于img * 0.4 + img2 * 0.6 + 0（这里的0是一个常数，可以随便改变）
img_all2 = cv2.addWeighted(img , 0.7 , img2 , 0.3 , 0)# 相当于img * 0.4 + img2 * 0.6 + 0（这里的0是一个常数，可以随便改变）
img_all3 = cv2.addWeighted(img , 0.7 , img2 , 0.3 , 100)# 相当于img * 0.4 + img2 * 0.6 + 0（这里的0是一个常数，可以随便改变）

plt.subplot(241) , plt.imshow(img_3x3y , 'gray') , plt.title('hight*3 wideth*3')# 原图3倍大小
plt.subplot(242) , plt.imshow(img_2x2y , 'gray') , plt.title('hight*2 wideth*2')# 原图2倍大小
plt.subplot(243) , plt.imshow(img_3x1y , 'gray') , plt.title('hight*1 wideth*3')# 三倍wideth，hight不变
plt.subplot(244) , plt.imshow(img_1x3y , 'gray') , plt.title('hight*3 wideth*1')# 三倍hight，wideth不变
plt.subplot(245) , plt.imshow(img_all , 'gray') , plt.title('直接add函数')# 直接相加
plt.subplot(246) , plt.imshow(img_all1 , 'gray') , plt.title('α = 0.3 β = 0.7')# α = 0.3 β = 0.7
plt.subplot(247) , plt.imshow(img_all2 , 'gray') , plt.title('α = 0.7 β = 0.3')# α = 0.7 β = 0.3
plt.subplot(248) , plt.imshow(img_all3 , 'gray') , plt.title('α = 0.7 β = 0.3 gamma = 100')# α = 0.7 β = 0.3 gamma = 100

plt.show()