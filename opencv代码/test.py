import cv2
import numpy as np

# 读取图像
image = cv2.imread('card math temple.png')

# 转换为灰度图像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 二值化处理
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# 查找轮廓
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 选择前两个轮廓
selected_contours = contours[:2]

# 绘制选择的轮廓
cv2.drawContours(image, selected_contours, -1, (0, 255, 0), 3)

# 显示结果
cv2.imshow('Selected Contours', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
