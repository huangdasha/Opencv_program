import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('Lina.png' , 0) # 读取灰度图像
img3 = cv2.imread('Lina.png' , cv2.IMREAD_COLOR) # 读取彩色图像
template = cv2.imread('Linas face.png' , 0)

# 匹配算法：
# TM_CCOEFF：计算相关性，计算出来的值越大，越相关
# TM_CCOEFF_NORMED：计算归一化相关性，计算出来的值越大，越相关
# TM_CCORR：计算相关性，计算出来的值越大，越相关
# TM_CCORR_NORMED：计算归一化相关性，计算出来的值越大，越相关
# TM_SQDIFF：计算平方差，计算出来的值越小，越相关
# TM_SQDIFF_NORMED：计算归一化平方差，计算出来的值越小，越相关
# 建议一般用归一化结果，归一化结果更准确。
methods = ['cv2.TM_CCOEFF' , 'cv2.TM_CCOEFF_NORMED' , 'cv2.TM_CCORR' , 'cv2.TM_CCORR_NORMED' , 'cv2.TM_SQDIFF' , 'cv2.TM_SQDIFF_NORMED']
res = cv2.matchTemplate(img , template , cv2.TM_SQDIFF)

# 输出的图像大小为：(图像的高度 - 模板的高度 + 1 , 图像的宽度 - 模板的宽度 + 1)
print("原图像大小为："  , img.shape)
print("模板图像大小为：" , template.shape)
print("模板匹配后输出的图像的大小为：" , res.shape)

for meth in methods:
    img2 = img3.copy()
    
    # 获取匹配算法的名称
    method = eval(meth)
    
    # 进行模板匹配
    res = cv2.matchTemplate(img , template , method)
    
    # 获取经过匹配的带的最大值和最小值以及其对应的位置
    min_val , max_val , min_loc , max_loc = cv2.minMaxLoc(res)
    
    # 如果是平方差匹配TM_SQDIFF或归一化平方差匹配TM_SQDIFF_NORMED，取最小值
    if method in [cv2.TM_SQDIFF , cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
        
    # 右下角点坐标 = 左上角点坐标 + 模板图像的宽高
    bottom_right = (top_left[0] + template.shape[1] , top_left[1] + template.shape[0])
    
    # 画出矩形(将矩形画在img2上大小通过左上和右下两个点来确定，且图像颜色为蓝色（2  255）)，且矩形变量为img2
    cv2.rectangle(img2 , top_left , bottom_right , 255 , 2)
    cv2.imshow("Output", img2)
    cv2.waitKey(0)
    