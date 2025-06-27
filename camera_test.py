import cv2
cap = cv2.VideoCapture(0)  # 0代表默认摄像头
if cap.isOpened():
    print("摄像头可用")
    ret, frame = cap.read()
    if ret:
        print(f"摄像头分辨率: {frame.shape}")
    else:
        print("无法读取摄像头画面")
else:
    print("无法打开摄像头")
cap.release()

for i in range(5):  # 检查前5个设备
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"摄像头 {i} 可用")
        cap.release()
    else:
        print(f"摄像头 {i} 不可用")