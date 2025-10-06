# README for Smart Shelf Monitor 

#### 1. 如何配置环境和运行程序的说明。

如何配置环境：本次任务使用运行于 VMware workstation pro 17 的 Ubuntu 22.04 系统完成，使用了`OpenCV`，`ultralytics`中的`yolov8n.pt`轻量级模型，以及`numpy`库完成。

系统依赖安装

  ```bash
  sudo apt update
  sudo apt install python3-opencv libgtk-3-dev v4l-utils -y
  ```

Python 依赖安装

  ```bash
  pip install ultralytics "numpy<2"
  ```
// # 这里注意，由于ultralytics 和 OpenCV 是编译在 NumPy 1.x 版本上的，所以需要安装的NumPy版本不得高于2.x版本。这个地方在最后排查了一个小时才发现😂。

如果出现了无法加载出相机画面，或者 `cv2.imshow()` 出现报错，说缺少图形界面支持（GTK）。
请执行：

  ```bash
  sudo apt install libgtk-3-dev
  ```

然后重新安装 OpenCV：

  ```bash
  pip uninstall opencv-python
  sudo apt install python3-opencv
  ```

这里 OpenCV 用的是 Ubuntu 官方编译的版本，带 GUI 支持即可解决无法弹窗的问题。

---

运行程序说明：

在运行程序之前，请检查电脑摄像头状态，确认摄像头是否可识别：

  ```bash
  v4l2-ctl --list-devices
  ```

若输出中包含 `/dev/video0`，说明驱动正常，或者也可下载并使用 `cheese` 测试画面。

请在终端输入 `v4l2-ctl --list-formats-ext` ，检查摄像头所支持的格式， `OpenCV` 默认支持的格式是 `YUYV`。

要执行项目，请先进入项目所在文件夹的终端。项目执行命令为

  ```bash
  python3 yolo_cam_detect_detailed.py 
  ```

>选择这个任务的时候我其实有点忐忑，毕竟我的Ubuntu是在VMware跑起来的，而虚拟机平台对摄像头硬件的支持很差，可能会有包括但不限于USB、驱动等问题。后面实操后也证明了这点，“该来的总会来的”。
>
>在整整一天时间的奋战中，我的代码和编译器和环境出现的情况包括但不限于虚拟机摄像头不兼容、GStreamer 不可用、OpenCV GUI 缺失、`NumPy` 版本不兼容。感谢 `Deepseek` 和 `ChatGPT` 以及各位开源的大佬们，代码最终实现了稳定的多目标检测与坐标输出。
>
>该项目实现了在虚拟机环境中通过 V4L2 驱动的 USB 摄像头进行实时 YOLOv8 目标检测。老实说，在没有多少 `OpenCV` 基础的情况下，根据 AI 的指挥敲代码实在是一件美逝。不过学习也就是这样，在曲折的道路上跌跌撞撞、增长经验、吸取教训。没有前面的艰辛，就不会有后面的果实。
>
>希望以后的代码都能顺利运行罢。

#### 2. 简述你的实现步骤。

1）初始化
	
导入库

```python
import cv2                      # OpenCV 用于图像处理和摄像头操作
import numpy as np              # NumPy 用于数值计算
from ultralytics import YOLO    # ultralytics 是 YOLOv8 的官方 Python 库
```

加载模型

```python
model = YOLO("yolov8n.pt")
```

使用 ultralytics 的 `YOLO("yolov8n.pt")` 加载模型。考虑到虚拟机可怜的性能，选择的是轻量级模型。🫠

2）初始化摄像头
 虚拟机中采用 V4L2 接口，设置格式为 YUYV，保证图像流可用：

```python
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
```

3）实时检测与绘制

在主循环中连续读取每一帧，并进行检测，防止 `Ubuntu` 读取失败（）

```python
while True:
    ret, frame = cap.read() # ret 是布尔值，表示是否读取成功， frame 是当前帧的图像 
    if not ret:
        print("无法读取摄像头画面。")
        break
```

循环读取帧、执行检测、绘制矩形框与中心点，并输出坐标：

```python
for result in results:
    for box in result.boxes:
        cls_name = model.names[int(box.cls[0])]
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
        cv2.putText(frame, f"{cls_name} ({cx},{cy})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        print(f"检测到 {cls_name}，中心点坐标: ({cx}, {cy})")
```

---



#### 3. **思考题**: 如果画面中同时出现多个水杯，你的程序会如何处理？如果需要获取物体在三维空间中的方向，仅靠当前信息足够吗？为什么？**（请至少在代码中实现对多个水杯情况的处理，并在README中详细阐述两种情况你的解决方案。）**

1）关于同时出现两个水杯的问题，因为我的程序在截取到的一帧中所有检测到的目标同等地识别并添加矩形框，然后每个框从 `yolov8n` 模型识别的名称数组中获取名字标签，这意味着不同识别到的物体，包括它的目标检测，命名，获取边框坐标，都是独立的过程。因此，完全不用担心出现两个同样类型的物体对 yolo 模型的识别造成影响，当识别到两个甚至多个同类物品时，呈现出来的仅仅是相同的标签，边界矩形框和中心点坐标完全不同。

2）如果需要依照本程序计算物体的三维空间的方向，除了需要读取二维图像（x,y图像）的深度信息（z轴长度），还需要读取物体在画面中的朝向。根据之前的线下培训内容，需要做出的改进有：采用双目摄像头、 RGB-D 摄像头（提供深度图），综合两个镜头读取到的视角差等和深度图等信息，使用 `cv2.solvePnP` 计算姿态。

#### 4. 项目结构：

  ```bash
  Smart_Shelf_Minitor/
  │
  ├── yolo_cam_detect_detailed.py   # 主程序
  ├── yolov8n.pt                    # 模型文件（首次运行自动下载）
  └── README.md                     # 项目说明文件
  ```