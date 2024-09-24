import numpy as np
import cv2
import matplotlib.pyplot as plt

# 与腐蚀操作相当于逆操作（腐蚀操作虽然可以消除毛刺但是会导致需要的图形变细，所以就可以用膨胀操作来使其尽量变回原样）
# 膨胀操作（原函数dst = cv2.dilate(src , kernel , iterations = x)）

img = cv2.imread('3.png' , cv2.IMREAD_UNCHANGED) # 读取灰度图像

# 先做腐蚀操作
kernel = np.ones((20 , 20) , np.uint8) # 定义一个5 * 5的卷积核，相当于卷积核内都为白色这个核就为白色否则就是黑色
erosion = cv2.erode(img , kernel , iterations = 1) # 腐蚀操作，iterations表示迭代次数，即腐蚀的程度

# 膨胀操作（将腐蚀操作后的图片传进去）
dilate = cv2.dilate(erosion , kernel , iterations = 1) # 膨胀操作，iterations表示迭代次数，即膨胀的程度
dilate2 = cv2.dilate(erosion , kernel , iterations = 2) # 2次膨胀操作，iterations表示迭代次数，即膨胀的程度
dilate3 = cv2.dilate(erosion , kernel , iterations = 3) # 3次膨胀操作，iterations表示迭代次数，即膨胀的程度

res = np.hstack((img , erosion , dilate)) # 将原图像，腐蚀操作后的图像，膨胀操作后的图像横向拼接起来
res2 = np.hstack((dilate , dilate2 , dilate3)) # 将原图像，腐蚀操作后的图像，膨胀操作后的图像横向拼接起来


# 下面两张图是原图->腐蚀图->膨胀图,以及膨胀迭代次数不同的图，可以明显看到膨胀操作的作用
cv2.imshow("result", res)# 膨胀操作后的图像

cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("result", res2)# 膨胀操作次数不同的图像

cv2.waitKey(0)
cv2.destroyAllWindows()
