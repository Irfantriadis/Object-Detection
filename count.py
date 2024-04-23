from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2

model = YOLO("model/yolov8m.pt")
cap = cv2.VideoCapture('data/jalan.mp4')
assert cap.isOpened()
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT,
                                       cv2.CAP_PROP_FPS))

region_of_interest = [(20, 600), (1700, 604), (1700, 560), (20, 560)]
# region_points = [(20, tinggi), (panjang, tinggi), (panjang, tinggi), (20, tinggi)]

# Video writer
video_writer = cv2.VideoWriter("object_counting_output.avi",
                       cv2.VideoWriter_fourcc(*'mp4v'),
                       fps,
                       (w, h))

# Init Object Counter
counter = object_counter.ObjectCounter()
counter.set_args(view_img=True,
                 reg_pts=region_of_interest,
                 classes_names=model.names,
                 draw_tracks=True)

frame_count = 0
max_frames = 10

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    tracks = model.track(im0, persist=True, show=False)

    im0 = counter.start_counting(im0, tracks)
    video_writer.write(im0)
    
    frame_count += 1

cap.release()
video_writer.release()
cv2.destroyAllWindows()