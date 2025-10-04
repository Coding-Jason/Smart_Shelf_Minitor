''' 导入所需库 '''
import cv2                      # OpenCV 用于图像处理和摄像头操作
import numpy as np              # NumPy 用于数值计算（例如坐标处理）
from ultralytics import YOLO    # ultralytics 是 YOLOv8 的官方 Python 库

''' 1. 加载预训练好的 YOLOv8 模型 '''
model = YOLO("yolov8n.pt")      # 加载 YOLOv8 的轻量级模型，速度快，准确率适中。

''' 2. YUYV 模式，使用 V4L2 后端直接打开摄像头 '''
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("无法打开摄像头！")
    exit()

print("恭喜你，成功打开摄像头！")

''' 3. 主循环：不断读取摄像头画面并检测 '''
while True:
    ret, frame = cap.read()                                     # ret 是布尔值，表示是否读取成功， frame 是当前帧的图像（numpy 数组）
    if not ret:
        print("无法读取摄像头画面。")
        break
    '''
    # DEBUGING 调试输出帧信息
    print("ret =", ret, ", frame type =", type(frame))
    if frame is not None:
        print("当前帧大小:", frame.shape)
    else:
        print("frame 是 None（空帧）")
    '''

    results = model.predict(frame, stream=True, verbose=False)  # 使用 YOLOv8 模型对这一帧进行目标检测

    ''' 4. 遍历每个检测结果 '''
    for result in results:
        for box in result.boxes:                                    # boxes 是该帧中所有检测到的目标框
            cls_id = int(box.cls[0])                                    # 获取类别名称
            cls_name = model.names[cls_id]
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)      # 获取边界框坐标
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)             # 计算中心点
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)    # 画出矩形框
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)             # 画出中心点
            #print(f"检测到 {cls_name}，中心点坐标: ({cx}, {cy})")          # 打印中心点坐标（终端）
            label = f"{cls_name} ({cx}, {cy})"                          # 在画面上显示类别和坐标
            cv2.putText(frame, label, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    ''' 5. 显示当前帧（带检测结果） '''
    cv2.imshow("YOLOv8 实时检测", frame)    # 创建一个窗口显示摄像头画面
    # cv2.waitKey(1) 等待 1 毫秒，并监听键盘输入
    # 如果按下 'q' 键，就退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

''' 6. 释放资源 '''
cap.release()           # 释放摄像头
cv2.destroyAllWindows() # 关闭所有 OpenCV 窗口
