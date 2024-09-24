import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('2.jpg',cv2.IMREAD_UNCHANGED) # 读取彩色图像
img_gray = cv2.imread('2.jpg',cv2.IMREAD_GRAYSCALE) # 读取灰度图像

# 将 BGR 图像转换为 RGB 格式(因为plt操作默认是RGB格式的，而cv2读取的是BGR格式的，所以需要转换一下)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 函数原型：retval, dst = cv2.threshold(src, thresh, maxval, type)
# ——dst输出图像，与输入图像有相同的大小和类型。
# ——src输入图像，只能输入单通道图像，通常来说为灰度图。
# ——thresh阈值。
# ——maxval当像素值超过了阈值（或者小于阈值，根据type来决定），所赋予的值。
# ——type二值化操作的类型，包含以下5种类型：
# cv2.THRESH_BINARY 超过阈值部分取maxval（最大值），否则取0
# cv2.THRESH_BINARY_INV THRESH_BINARY的反转（超过阈值取0反之max）
# cv2.THRESH_TRUNC 大于阈值部分设为阈值(如大于127部分便为127)，否则不变 
# cv2.THRESH_TOZERO 大于阈值部分不改变，否则设为0
# cv2.THRESH_TOZERO_INV THRESH_TOZERO的反转

# 以下便是五个不同的阈值处理的示例，这个函数会返回两个值，第一个是阈值（ret），第二个是处理后的图像（thresh1）
ret , thresh1 = cv2.threshold(img_gray, 127 , 255 , cv2.THRESH_BINARY)
ret , thresh2 = cv2.threshold(img_gray, 127 , 255 , cv2.THRESH_BINARY_INV)
ret , thresh3 = cv2.threshold(img_gray, 127 , 255 , cv2.THRESH_TRUNC)
ret , thresh4 = cv2.threshold(img_gray, 127 , 255 , cv2.THRESH_TOZERO)
ret , thresh5 = cv2.threshold(img_gray, 127 , 255 , cv2.THRESH_TOZERO_INV)

titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
images = [img_rgb, thresh1, thresh2, thresh3, thresh4, thresh5]

# 取i0~5，再i+1形式利用plt.subplot()函数将多个图像显示在一个窗口中
for i in range(6):
    # 2行3列的图形，当前位置在第i+1个位置（1~6）
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
    
    # 设置标题
    plt.title(titles[i])
    
    # 不显示坐标轴
    plt.xticks([]),plt.yticks([])

plt.show()