import torch
import pygame
import tensorflow as tf
import argparse
import cv2 , numpy as np , pandas as pd
import tensorflow as tf
from re import DEBUG ,sub
from flask import Flask, render_template , request , redirect, send_file , url_for, Response
from werkzeug.utils import secure_filename , send_from_directory
import os , subprocess
from subprocess import Popen
import re , requests ,shutil , time , glob
from PIL import Image
from ultralytics import YOLO
from keras.preprocessing.image import ImageDataGenerator
import io
import base64
app = Flask(__name__  )
@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/<path:filename>')
def display(filename):
    
    folder_path = r'C:\Users\siddh\Desktop\RoadSense-Intelligent-Road-Monitoring-System-for-Drivers-assistance-\Web_GUI\runs\detect'
    subfolder = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    latest_subfolder = max(subfolder, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
    directory = folder_path+"/"+latest_subfolder
    print("printing directory : --------"  , directory)
    files = os.listdir(directory)
    latest_file  = files[0]
    print("printing latest_file : --------"  , latest_file)
   
    filename = os.path.join(folder_path,latest_subfolder, latest_file)
    file_extension = filename.rsplit('.',1)[1].lower()
    environ = request.environ
    img = cv2.imread(filename)
    print("printing file name :============",filename)
    _, buffer = cv2.imencode('.jpg', img)
    image_data = base64.b64encode(buffer).decode('utf-8')

    return render_template('index.html', image_data=image_data)
    if file_extension == 'jpg':
        # return send_from_directory(directory , latest_file , environ)
        return render_template('index.html', image_path=filename)
    
    else:
        return "Invalid file format"


def get_frame():
    folder_path = os.getcwd()
    mp4_files = 'output.mp4'
    video = cv2.VideoCapture(mp4_files)
    while True:
        success , image = video.read()
        if not success:
            break
        ret, jpgg = cv2.imencode('.jpg', image)
        yield(b'--frame\r\n'
                b'Content-Type: image/jpgg\r\n\r\n' + jpgg.tobytes() + b'r\n\r\n')
        # time.sleep(0,1)

@app.route("/video_feed")
def video_feed():
    print("function called")
    return Response(get_frame(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/", methods=["GET", "POST"])
def predict_img():
    pygame.init()
    

    if request.method == 'POST':
        if 'file' in request.files:
            f = request.files['file']
            basepath = os.path.dirname(__file__)
            filepath = os.path.join(basepath,'uploads',f.filename)
            print("Uploaded folder is " , filepath)

            # filepath =filepath+".jpg"
            f.save(filepath)
            global imgpath
            predict_img.imgpath = f.filename
            
            print("Printing predict_img :::::" , predict_img , f.filename)

            file_extension = f.filename.rsplit('.',1)[1].lower()
            print("-------------------",file_extension , filepath)
            if file_extension == 'jpg':
                img = cv2.imread(filepath)
                _, frame = cv2.imencode('.jpg', img)
                image = Image.open(io.BytesIO(frame))
# increasing the quality of image
                datagen = ImageDataGenerator(brightness_range=[0.5,2.0])
                for x in range(image.size[0]):
                    for y in range(image.size[1]):
                        r, g, b = image.getpixel((x, y))
                        image.putpixel((x, y), (r, g, 0))

                yolo = YOLO('bestv2.pt')
                print("detection about to start")
                print(f.filename)
                # print(type(image))
                detections = yolo.predict(filepath ,save = True )

                # Assuming detection[0] is the Results object
                results = detections[0]  # Assuming detection[0] is the Results object
                class_names  = results.probs # Get the dictionary of class names
                class_ = results.boxes.cls
                if results.boxes.cls.size(0) != 0:
                    for value in class_:
                        value = int(value.item())
                
                    if value == 0:
                        print("Speed Breaker")
                        pygame.mixer.music.load('breaker.wav')
                        pygame.mixer.music.play()
                    elif value == 1:
                        print("crosswalk")
                        pygame.mixer.music.load('speed.mp3')
                        pygame.mixer.music.play()
                    elif value == 3:
                        print("speedlimit")
                        pygame.mixer.music.load('speed.mp3')
                        pygame.mixer.music.play()
                    elif value == 4:
                        print("stop")
                        pygame.mixer.music.load('stop_sound.wav')
                        pygame.mixer.music.play()

                    elif value == 5:
                        print("trafficlight")
                    return display(f.filename)
                else:
                    return display(f.filename)
                print("")
                print("")
                
            elif file_extension == 'mp4':
                video_path  = filepath
                cap = cv2.VideoCapture(video_path)
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter('output.mp4' , fourcc, 30.0,(frame_width,frame_height))

                model = YOLO('bestv2.pt')
                frame_counter = 0
                while cap.isOpened():
                    ret , frame = cap.read()
                    if not ret:
                        break
                    
                    if frame_counter % 8 == 0:
                        results = model(frame , save= True)
                        print(results)
                        cv2.waitKey(1)

                        res_plotted = results[0].plot()
                        cv2.imshow("result", res_plotted)
                    frame_counter +=1
                    out.write(res_plotted)


                    if cv2.waitKey(1) == ord('q'):
                        break
                return video_feed()


    


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description = "Flask app exposing yolov8 models")
    parser.add_argument("--port" , default = 5500, type = int , help = "port number")
    args = parser.parse_args()
    # model = torch.hub.load('.', 'custom','best.pt', source='local')
    # model.eval()
    app.run(host="127.0.0.1", port=args.port , debug = False) 