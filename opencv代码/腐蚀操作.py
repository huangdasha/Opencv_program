import numpy as np
import cv2
import matplotlib.pyplot as plt

# 腐蚀操作是对于只有两个值的图像（即黑白图像）来说的，对于彩色图像来说，是没有腐蚀操作的
# 其原理就是看每个像素点内是否全为白色，若有一个点不是白色，则将这个点变为黑色，这样就达到一个腐蚀的效果
# kernel:卷积核
# iterations:迭代次数，即腐蚀的程度
# 腐蚀操作（原函数dst = cv2.erode(src , kernel , iterations = x)）
img = cv2.imread('3.png' , cv2.IMREAD_UNCHANGED) # 读取灰度图像

# 定义一个5 * 5的卷积核
kernel = np.ones((20 , 20) , np.uint8) # 相当于卷积核内都为白色这个核就为白色否则就是黑色
erosion = cv2.erode(img , kernel , iterations = 1) # 腐蚀操作，iterations表示迭代次数，即腐蚀的程度

#显示图像
cv2.imshow("src", img)# 原图像
cv2.imshow("result", erosion)# 腐蚀操作后的图像
 
#等待显示
cv2.waitKey(0)
cv2.destroyAllWindows()