import numpy as np
import cv2
import matplotlib.pyplot as plt

def img_show(img , title):
    cv2.namedWindow(title, cv2.WINDOW_NORMAL) # 设置窗口标题
    cv2.resizeWindow(title, 300, 250)  # 你可以根据需要设置不同的宽度和高度
    cv2.imshow(title , img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 主要是利用一个图像轮廓的检测函数
# cv.findContours函数（查找轮廓）：cv2.findContours(image, mode, method)
# ——image：输入图像，只能输入单通道图像，通常来说为灰度图。（最好是经过二值法处理的灰度图像）
# ——mode：轮廓检索模式
# cv2.RETR_EXTERNAL：只检测外轮廓
# cv2.RETR_LIST：检索所有的轮廓，并保存到一个链表之中。
# cv2.RETR_CCOMP：检索所有的轮廓，并将他们分为两个等级的轮廓。上一层是外部边界，里一层是空洞的边界。
# cv2.RETR_TREE：检索所有轮廓，并建立一个等级树结构的轮廓。（一般只用这一个检索模式）

# ——method：轮廓逼近方法
# cv2.CHAIN_APPROX_NONE：以Freeman链码的方式输出轮廓，所有其他方法输出多边形（顶点的序列）
# cv2.CHAIN_APPROX_SIMPLE：压缩水平的、垂直的和斜的部分，也就是，函数只保留他们的终点部分（就是保留一些关键点）
# 一般是使用上面两种轮廓逼近的方法。
# cv2.CHAIN_APPROX_TC89_L1：使用teh-Chinl chain 近似算法
# cv2.CHAIN_APPROX_TC89_KCOS：使用teh-Chinl chain 近似算法

# 返回值有两个：contours（轮廓）, hierarchy（层级）
img = cv2.imread('6.png') # 读取彩色图像(用来当作后面画轮廓的原图像)    
img2 = cv2.imread('10.png') # 读取彩色图像(用来当作后面画轮廓的原图像)
img_gray = cv2.imread('6.png',cv2.IMREAD_GRAYSCALE) # 读取灰度图像（这个是lina（女性角色）的图片）
img2_gray = cv2.imread('10.png' , cv2.IMREAD_GRAYSCALE) # 读取灰度图像(这个是那个很多规则图案的图片)

# 将两张图片进行二值化处理（将值全部转化为255和0只有两种颜色）
# 函数原型：cv2.threshold(src, thresh, maxval, type, dst=None)
ret , thresh = cv2.threshold(img_gray , 127 , 255 , cv2.THRESH_BINARY)
img_show(thresh , 'thresh')
ret2 , thresh2 = cv2.threshold(img2_gray , 127 , 255 , cv2.THRESH_BINARY)
img_show(thresh2 , 'thresh2')

# 传进来的是一个经过二值计算的灰度图，contours就是我们所需要的轮廓
contours , hierarchy = cv2.findContours(thresh , cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)
contours2 , hierarchy2 = cv2.findContours(thresh2 , cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)


# 画出轮廓(相当于把轮廓画在原图像上,注意画的时候会改变原图像的值，所以最好是先复制一份原图像)
# 函数原型：cv2.drawContours(image, contours, contourIdx, color, thickness)
# ——image：原图像
# ——contours：轮廓本身，是一个list，list中每个元素都是一个轮廓点集（不是每个轮廓点集都是一个闭合的轮廓）
# ——contourIdx：轮廓的索引（等于是画第几个轮廓，-1则为画所有的轮廓）
# ——color：轮廓的颜色。
# ——thickness：轮廓的粗细。
draw_img = img.copy()# 这里是复制一份原图像，是因为画轮廓的时候会改变原图像的值，所以复制一份可以保证原图像不被破坏。
ret = cv2.drawContours(draw_img , contours , -1 , (0 , 0 , 255) , 2)
img_show(ret , 'ret')

draw_img2 = img2.copy()# 这里是复制一份原图像，是因为画轮廓的时候会改变原图像的值，所以复制一份可以保证原图像不被破坏。
ret2 = cv2.drawContours(draw_img2 , contours2 , -1 , (0 , 0 , 255) , 2)
img_show(ret2 , 'ret2')

# 轮廓特征
# 首先要计算轮廓的特征需要把第一个轮廓提取出来随后再利用函数进行计算。
cnt = contours[0] # 提取出第一个轮廓（可以理解为把轮廓第一个节点的地址给出来（因为contours得到的是返回的一个list类型的变量））
cnt2 = contours2[0] # 提取出第一个轮廓（可以理解为把轮廓第一个节点的地址给出来（因为contours得到的是返回的一个list类型的变量））
# 计算轮廓面积函数原型：cv2.contourArea(contours)
print(cv2.contourArea(cnt2))
# 计算轮廓周长函数原型：cv2.arcLength(contours , True) 
print(cv2.arcLength(cnt , True)) # True表示轮廓是闭合的，False表示轮廓是不闭合的


# 轮廓的近似
# 函数原型：cv2.approxPolyDP(curve, epsilon, closed)
# ——curve：输入的点集
# ——epsilon：指定的精度，也就是原始曲线与近似曲线之间的最大距离。
# ——closed：曲线是否是闭合的
img3 = cv2.imread('11.png') # 读取彩色图像(用来当作后面画轮廓的原图像)
img3_gray = cv2.imread('11.png',cv2.IMREAD_GRAYSCALE) # 读取灰度图像

ret3 , thresh3 = cv2.threshold(img3_gray , 100 , 255 , cv2.THRESH_BINARY)# 二值化处理
contours3 , hierarchy3 = cv2.findContours(thresh3 , cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)# 查找轮廓

cnt3 = contours3[0] # 提取出第一个轮廓

eplision = 0.01 * cv2.arcLength(cnt3 , True) # 这里是指定的精度，也就是原始曲线与近似曲线之间的最大距离.
approx = cv2.approxPolyDP(cnt3 , eplision , True) # 这里是对轮廓进行近似处理.

img_draw3 = img3.copy()# 这里是复制一份原图像，是因为画轮廓的时候会改变原图像的值，所以复制一份可以保证原图像不被破坏。
res3 = cv2.drawContours(img_draw3, [approx], -1, (0, 0, 255), 3)
img_show(res3 , 'res3')
