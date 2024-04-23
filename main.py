from ultralytics import YOLO

model = YOLO('model/yolov8m.pt')

data = 'data/jalan.mp4'
model.predict(source=data, show=True, conf=0.5)