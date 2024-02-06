import cv2 , numpy as np 
from PIL import Image
from ultralytics import YOLO
import io
print("hui")
filepath = r"D:\PROJECTS\DIP\Web_GUI\uploads\newtwoonweee (19).jpg"
img = cv2.imread(filepath)
_, frame = cv2.imencode('.jpg', img)
image = Image.open(io.BytesIO(frame))
yolo = YOLO('best.pt')
results = yolo.predict(img , save = True)
# processed_image = results[0].orig_img
# cv2.imwrite('output.jpg', processed_image)
print("hui hui")