import cv2
import numpy as np

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH , 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT , 240)
while(1):
    # 读取视频流转化为一帧一帧的图片
    ret , frame = cap.read()

    # 转换为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 二值化处理
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # 轮廓检测
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 遍历轮廓
    for contour in contours:
        # 获取轮廓的矩形边界框
        x, y, w, h = cv2.boundingRect(contour)
    
    # 过滤掉太小的轮廓
    if w * h < 100:
        continue
    
    # 在原始图像上绘制边界框
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 显示结果
    cv2.imshow('Digits Detected', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
