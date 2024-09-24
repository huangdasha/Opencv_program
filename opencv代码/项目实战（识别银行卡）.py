import numpy as np
import cv2
import matplotlib.pyplot as plt
from imutils import contours
import imutils
import myutils

def cv2_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img = cv2.imread('card math temple.png') 
ref2 = cv2.imread('card math temple.png' , cv2.IMREAD_GRAYSCALE) # 读取灰度图像
ref = ref2[2:110 , 2:915]

print(ref.shape)

# 函数原型：cv2.threshold(src, thresh, maxval, type)
# ——src：输入图，只能输入单通道图，一般为灰度图
# ——thresh：阈值
# ——maxval：当像素值超过了阈值（或者小于阈值，根据type来决定），所赋予的值
# ——type：二值化操作的类型
# cv2.THRESH_BINARY：超过阈值部分取maxval（最大值），否则取0
# cv2.THRESH_BINARY_INV：THRESH_BINARY的反转
# cv2.THRESH_TRUNC：大于阈值部分设为阈值，否则不变
# cv2.THRESH_TOZERO：大于阈值部分不改变，否则设为0
# cv2.THRESH_TOZERO_INV：THRESH_TOZERO的反转
ref = cv2.threshold(ref , 10 , 255 , cv2.THRESH_BINARY)[1] # 将模板图像进行二值化处理
cv2_show('after two-in-data' , ref)

# 函数原型：cv2.findContours(image, mode, method)
# ——image：输入图像，只能输入单通道图像，通常来说为灰度图。（最好是经过二值法处理的灰度图像）
# ——mode：轮廓检索模式
# cv2.RETR_EXTERNAL：只检测外轮廓
# cv2.RETR_LIST：检索所有的轮廓，并保存到一个链表之中。
# cv2.RETR_CCOMP：检索所有的轮廓，并将他们分为两个等级的轮廓。上一层是外部边界，里一层是空洞的边界。
# cv2.RETR_TREE：检索所有轮廓，并建立一个等级树结构的轮廓。（一般只用这一个检索模式）
# ——method：轮廓逼近方法
# cv2.CHAIN_APPROX_NONE：以Freeman链码的方式输出轮廓，所有其他方法输出多边形（顶点的序列）
# cv2.CHAIN_APPROX_SIMPLE：压缩水平的、垂直的和斜的部分，也就是，函数只保留他们的终点部分（就是保留一些关键点）
refCnts , hierarchy = cv2.findContours(ref.copy() , cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)
# 对轮廓按周长进行排序
refCnts = sorted(refCnts , key = cv2.contourArea , reverse = True)
# 其中0是周长最大的轮廓，[1:6]是选择轮廓1~5,[7:12]是选择轮廓7~11，所以一共十个轮廓，是所有数字的外围轮廓。
refCnts = refCnts[1:6] + refCnts[7:12] # 排除第一个（周长最大）轮廓
# 画出模板图像的轮廓
img_copy = img.copy()
cv2.drawContours(img_copy , refCnts , -1 , (0 , 0 , 255) , 3)
cv2_show('after drawContours()' , img_copy)


# 获取每个轮廓的边界框并存储左上角坐标
bounding_boxes = []
for selected_refCnt in refCnts:
    x, y, w, h = cv2.boundingRect(selected_refCnt)
    bounding_boxes.append((x, y))
# 按照边界框的左上角坐标对轮廓进行排序
# 这里 [x for x, _ in bounding_boxes] 是一个列表推导式，用于从 bounding_boxes 中提取每个边界框的 x 坐标。
# np.argsort([x for x, _ in bounding_boxes]) 返回按照 x 坐标排序后的索引。
refCnts = [refCnts[i] for i in np.argsort([x for x, _ in bounding_boxes])] # 这样所有的数字就是按照0~9的顺序排列了

degits = {}
# 遍历每一个排序过后的轮廓，其中i是索引，c是轮廓
# enumerate()函数接受一个可迭代对象作为参数，并返回一个由索引-值对组成的枚举对象。
# 在这个枚举对象中，每个索引-值对都表示可迭代对象中对应元素的索引和值。而索引对应的值赋给了i，结果给了c。
for (i , c) in enumerate(refCnts):
    (x , y , w , h) = cv2.boundingRect(c)
    roi = ref[y : y + h , x : x + w]
    roi = cv2.resize(roi , (57 , 88)) 
    # 每一个数字对应一个模板
    degits[i] = roi
    # cv2_show('roi' , roi)

# 初始化卷积核（为了防止除账号以外的干扰项）
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT , (9 , 3)) # 9*3的矩形卷积核
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT , (5 , 5)) # 5*5的矩形卷积核

# 读取银行卡图像并且修改大小
imgae = cv2.imread('bank card.png')
imgae = cv2.resize(imgae , (300 , 181))
cv2_show('original imgae' , imgae)

# 主要是为了突出英航卡明亮的区域
gray = cv2.cvtColor(imgae , cv2.COLOR_BGR2GRAY)# 转换为灰度图像
tophat = cv2.morphologyEx(gray , cv2.MORPH_TOPHAT , rectKernel) # 顶帽操作，突出图像的明亮区域
cv2_show('after tophat' , tophat) 

# Sobel算子求x方向的梯度，并进行取绝对值和归一化操作用处是为了突出银行卡号的区域
gradX = cv2.Sobel(tophat , ddepth = cv2.CV_32F , dx = 1 , dy = 0 , ksize = -1) # 求x方向的梯度
gradX = np.absolute(gradX) # 取绝对值
(minVal , maxVal) = (np.min(gradX) , np.max(gradX))# 求最大值和最小值
gradX = (255 * ((gradX - minVal) / (maxVal - minVal))) # 归一化
print(gradX.astype)
gradX = gradX.astype('uint8') # 转换数据类型
cv2_show('gradX' , gradX)

# 通过闭操作（先膨胀，后腐蚀）将数字连在一起
gradX = cv2.morphologyEx(gradX , cv2.MORPH_CLOSE , rectKernel) # 闭操作
cv2_show('after close_step gradX' , gradX)   

# 这里阈值为0是因为不确定阈值范围，而上面用了归一化所以只有两种的主体，所以用OTSU二值化
# THRESH_BINARY：超过阈值部分取maxval（最大值），否则取0
# THRESH_OTSU：采用OTSU算法自动寻找全局阈值（适合双峰（两种主体的））
thresh = cv2.threshold(gradX , 0 , 255 , cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1] # 二值化，只剩下黑白了，为了后面对数字的定位
cv2_show('after thresh' , thresh)

# 再来一次闭操作（先膨胀，后腐蚀）将数字之间间隙部分填充上。
thresh = cv2.morphologyEx(thresh , cv2.MORPH_CLOSE , rectKernel) # 闭操作
cv2_show('after close_step gradX' , thresh)

# 计算轮廓
# 函数原型：cv2.findContours(image, mode, method)
# ——image：输入图像，只能输入单通道图像，通常来说为灰度图。（最好是经过二值法处理的灰度图像）
# ——mode：轮廓检索模式
# cv2.RETR_EXTERNAL：只检测外轮廓
# cv2.RETR_LIST：检索所有的轮廓，并保存到一个链表之中。
# cv2.RETR_CCOMP：检索所有的轮廓，并将他们分为两个等级的轮廓。上一层是外部边界，里一层是空洞的边界。
# cv2.RETR_TREE：检索所有轮廓，并建立一个等级树结构的轮廓。（一般只用这一个检索模式）
# ——method：轮廓逼近方法
# cv2.CHAIN_APPROX_NONE：以Freeman链码的方式输出轮廓，所有其他方法输出多边形（顶点的序列）
# cv2.CHAIN_APPROX_SIMPLE：压缩水平的、垂直的和斜的部分，也就是，函数只保留他们的终点部分（就是保留一些关键点）
cnts , hierarchy = cv2.findContours(thresh.copy() , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
cur_img = imgae.copy()
cv2.drawContours(cur_img , cnts , -1 , (0 , 0 , 255) , 3)
cv2_show('after findContours()' , cur_img)

locs = []
# 遍历轮廓
for (i , c) in enumerate(cnts):
    # 计算矩形
    (x , y , w , h) = cv2.boundingRect(c)
    ar = w / float(h) # 计算宽高比
    # 选择合适的区域，根据实际任务来，这里的基本都是四个数字的区域
    if ar > 2.5 and ar < 4.0:
        if (w > 40 and w < 55) and (h > 10 and h < 20):
            # 符合的留下来
            locs.append((x , y , w , h))
locs = sorted(locs , key = lambda x : x[0]) # 根据x坐标进行排序

output = []
# 遍历每一个轮廓中的数字
for (i , (gX , gY , gW , gH)) in enumerate(locs):
    groupOutput = []
    # 根据坐标提取每一个组(加5啥的是为了扩大范围)
    group = gray[gY - 5 : gY + gH + 5 , gX - 5 : gX + gW + 5]
    cv2_show('group' , group)# 将每个组展示出来
    # 预处理（二值化并取自适应阈值）
    group = cv2.threshold(group , 0 , 255 , cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv2_show('group' , group)# 将每个组经过二值化得到的展示出来
    # 计算每一组的轮廓
    groupCnts , hierarchy = cv2.findContours(group.copy() , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
    groupCnts = contours.sort_contours(groupCnts , method = 'left-to-right')[0]
    # 对每组进行计算，计算得到每一个数字
    for c in groupCnts:
        (x , y , w , h) = cv2.boundingRect(c)
        roi = group[y : y + h , x : x + w]
        roi = cv2.resize(roi , (57 , 88))# 这里的(57,88)是因为之前我们截取的数字模板就是这个大小
        # 计算匹配得分
        scores = []
        # 在模板中计算每一个得分
        # 用于遍历一个字典 digits 中的键值对（key-value pairs）。在每次迭代中，digit 将代表字典中的键，digitROI 将代表对应键所对应的值。
        for (digit , digitROI) in degits.items():
            result = cv2.matchTemplate(roi , digitROI , cv2.TM_SQDIFF)# 跟之前获取的数字模板进行模板匹配
            (_, score , _ , _) = cv2.minMaxLoc(result)# 返回最大值和最小值的坐标
            scores.append(score)# 将得分放到列表中
        # 得到最合适的数字，这里最合适的数字就是得分最高的
        groupOutput.append(str(np.argmax(scores)))
        
    # 画出来
    cv2.rectangle(imgae , (gX - 5 , gY - 5) , (gX + gW + 5 , gY + gH + 5) , (0 , 0 , 255) , 1)
    cv2.putText(imgae , ''.join(groupOutput) , (gX , gY - 15) , cv2.FONT_HERSHEY_SIMPLEX , 0.65 , (0 , 0 , 255) , 2)
    # 得到结果
    output.extend(groupOutput)
cv2_show('imgae' , imgae)   