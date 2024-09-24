import numpy as np
import cv2
import matplotlib.pyplot as plt

# 礼帽：原始输入 - 开运算结果(突出毛刺部分)
# 黑帽：闭运算结果 - 原始输入（突出轮廓部分）
img = cv2.imread('4.png' , cv2.IMREAD_UNCHANGED) # 读取灰度图像
kernel = np.ones((20 , 20) , np.uint8) # 定义一个5 * 5的卷积核，相当于卷积核内都为白色这个核就为白色否则就是黑色

# 礼帽操作(dst = cv2.morphologyEx(src , cv2.MORPH_TOPHAT , kernel)),即原始输入 - 开运算结果
tophat = cv2.morphologyEx(img , cv2.MORPH_TOPHAT , kernel) # 礼帽操作，即原始输入 - 开运算结果

# 黑帽操作(dst = cv2.morphologyEx(src , cv2.MORPH_BLACKHAT , kernel),即闭运算结果 - 原始输入
blackhat = cv2.morphologyEx(img , cv2.MORPH_BLACKHAT , kernel) # 黑帽操作，即闭运算结果 - 原始输入

# 将几张图象横向放在一起显示
all = np.hstack((img , tophat , blackhat)) # 将原图像，礼帽操作后的图像，黑帽操作后的图像横向拼接起来

# 设置窗口大小
cv2.namedWindow('result', cv2.WINDOW_NORMAL) # 设置窗口标题
cv2.resizeWindow('result', 1600, 600)  # 你可以根据需要设置不同的宽度和高度

cv2.imshow("result", all)# 膨胀操作后的图像
cv2.waitKey(0)
cv2.destroyAllWindows()
