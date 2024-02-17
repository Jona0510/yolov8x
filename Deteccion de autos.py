from ultralytics import YOLO
from ultralytics.solutions import object_counter
import matplotlib.pyplot as plt
import cv2

model = YOLO("yolov8x.pt")
cap = cv2.VideoCapture("Video1.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

#--------------Deteccion de nuestro primer pixel para trazar nuestro contador---#
# # Leer el primer fotograma
# success, im0 = cap.read()
# if not success:
#     print("Unable to read video frame")
# # Mostrar el primer fotograma
# plt.imshow(cv2.cvtColor(im0, cv2.COLOR_BGR2RGB))
# plt.title('First Frame')
# plt.show()
# # Define region points
region_points = [(200, 500), (1200, 500)] #1 -----   #2 ----- 1

# Video writer
video_writer = cv2.VideoWriter("Resultado1.mp4",
                       cv2.VideoWriter_fourcc(*'mp4v'),
                       fps,
                       (w, h))

# Init Object Counter
counter = object_counter.ObjectCounter()
counter.set_args(view_img=True,
                 reg_pts=region_points,
                 classes_names=model.names,
                 draw_tracks=False)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("No se encontraron mas fotogramas o el video fue procesado completamente")
        break
    tracks = model.track(im0, persist=True, show=False,classes=[2,3,7],tracker="bytetrack.yaml")
 
    im0 = counter.start_counting(im0, tracks)
    video_writer.write(im0)             
    
cap.release()
video_writer.release()
cv2.destroyAllWindows()