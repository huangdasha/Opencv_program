import numpy as np
import cv2
import matplotlib.pyplot as plt

top_size , bottom_size , left_size , right_size = (50 , 50 , 50 , 50)# 设置待会要使用填充边界的大小的值

img = cv2.imread('1.jpg' , cv2.IMREAD_COLOR) # 读取彩色图像(RGB图像)

# 将 BGR 图像转换为 RGB 格式(因为plt操作默认是RGB格式的，而cv2读取的是BGR格式的，所以需要转换一下)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 以下是几种不同的图片填充方式(前面的都一样，只有最后一个type（复制样式）不同)
replicate = cv2.copyMakeBorder(img_rgb , top_size , bottom_size , left_size , right_size , borderType = cv2.BORDER_REPLICATE)# 复制边界
reflect = cv2.copyMakeBorder(img_rgb , top_size , bottom_size , left_size , right_size , borderType = cv2.BORDER_REFLECT)# 反射边界 abcd | dcba
reflect101 = cv2.copyMakeBorder(img_rgb , top_size , bottom_size , left_size , right_size , borderType = cv2.BORDER_REFLECT_101)# 反射边界 abcd | cba
wrap = cv2.copyMakeBorder(img_rgb , top_size , bottom_size , left_size , right_size , borderType = cv2.BORDER_WRAP)# 外包装边界
constant = cv2.copyMakeBorder(img_rgb , top_size , bottom_size , left_size , right_size , borderType = cv2.BORDER_CONSTANT , value = 0)# 常数边界value代表边界填充的常数

# 以下是生成一个拥有多个子图的的图形
# 第一列代码中的231代表创建一个2行3列的图形，并且当前位置在第1个位置
# 第二列代码中的‘gray’表示图像以灰度形式显示(可以随便输入一个错误的看报错再选择要什么形式的（报错会提供所有可选择的选项）)
plt.subplot(231) , plt.imshow(img_rgb , 'gray') , plt.title('ORIGINAL')# 原图
plt.subplot(232) , plt.imshow(replicate , 'gray') , plt.title('REPLICATE')# 复制边界
plt.subplot(233) , plt.imshow(reflect , 'gray') , plt.title('REFLECT')# 反射边界
plt.subplot(234) , plt.imshow(reflect101 , 'gray') , plt.title('REFLECT_101')# 反射边界
plt.subplot(235) , plt.imshow(wrap , 'gray') , plt.title('WRAP')# 外包装边界
plt.subplot(236) , plt.imshow(constant , 'gray') , plt.title('CONSTANT')# 常数边界

plt.show()