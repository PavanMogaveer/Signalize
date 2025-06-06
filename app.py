# from __future__ import division, print_function
# import sys
# import os
# import glob
# import re
# import numpy as np
# import tensorflow as tf
# import tensorflow as tf
# import cv2

# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image

# from flask import Flask, redirect, url_for, request, render_template
# from werkzeug.utils import secure_filename

# app = Flask(__name__)

# MODEL_PATH ='model.h5'

# model = load_model(MODEL_PATH)

# def grayscale(img):
#     img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     return img
# def equalize(img):
#     img =cv2.equalizeHist(img)
#     return img
# def preprocessing(img):
#     img = grayscale(img)
#     img = equalize(img)
#     img = img/255
#     return img
# def getClassName(classNo):
#     if   classNo == 0: return 'Speed Limit 20 km/h'
#     elif classNo == 1: return 'Speed Limit 30 km/h'
#     elif classNo == 2: return 'Speed Limit 50 km/h'
#     elif classNo == 3: return 'Speed Limit 60 km/h'
#     elif classNo == 4: return 'Speed Limit 70 km/h'
#     elif classNo == 5: return 'Speed Limit 80 km/h'
#     elif classNo == 6: return 'End of Speed Limit 80 km/h'
#     elif classNo == 7: return 'Speed Limit 100 km/h'
#     elif classNo == 8: return 'Speed Limit 120 km/h'
#     elif classNo == 9: return 'No passing'
#     elif classNo == 10: return 'No passing for vechiles over 3.5 metric tons'
#     elif classNo == 11: return 'Right-of-way at the next intersection'
#     elif classNo == 12: return 'Priority road'
#     elif classNo == 13: return 'Yield'
#     elif classNo == 14: return 'Stop'
#     elif classNo == 15: return 'No vechiles'
#     elif classNo == 16: return 'Vechiles over 3.5 metric tons prohibited'
#     elif classNo == 17: return 'No entry'
#     elif classNo == 18: return 'General caution'
#     elif classNo == 19: return 'Dangerous curve to the left'
#     elif classNo == 20: return 'Dangerous curve to the right'
#     elif classNo == 21: return 'Double curve'
#     elif classNo == 22: return 'Bumpy road'
#     elif classNo == 23: return 'Slippery road'
#     elif classNo == 24: return 'Road narrows on the right'
#     elif classNo == 25: return 'Road work'
#     elif classNo == 26: return 'Traffic signals'
#     elif classNo == 27: return 'Pedestrians'
#     elif classNo == 28: return 'Children crossing'
#     elif classNo == 29: return 'Bicycles crossing'
#     elif classNo == 30: return 'Beware of ice/snow'
#     elif classNo == 31: return 'Wild animals crossing'
#     elif classNo == 32: return 'End of all speed and passing limits'
#     elif classNo == 33: return 'Turn right ahead'
#     elif classNo == 34: return 'Turn left ahead'
#     elif classNo == 35: return 'Ahead only'
#     elif classNo == 36: return 'Go straight or right'
#     elif classNo == 37: return 'Go straight or left'
#     elif classNo == 38: return 'Keep right'
#     elif classNo == 39: return 'Keep left'
#     elif classNo == 40: return 'Roundabout mandatory'
#     elif classNo == 41: return 'End of no passing'
#     elif classNo == 42: return 'End of no passing by vechiles over 3.5 metric tons'


# def model_predict(img_path, model):
#     print(img_path)
#     img = image.load_img(img_path, target_size=(224, 224))
#     img = np.asarray(img)
#     img = cv2.resize(img, (32, 32))
#     img = preprocessing(img)  # Assuming this is your custom preprocessing function
#     cv2.imshow("Processed Image", img)
#     img = img.reshape(1, 32, 32, 1)

#     # Predict image
#     predictions = model.predict(img)
#     classIndex = np.argmax(predictions, axis=1)[0]

#     preds = getClassName(classIndex)
#     return preds



# @app.route('/', methods=['GET'])
# def index():
#     # Main page
#     return render_template('index.html')


# @app.route('/predict', methods=['GET', 'POST'])
# def upload():
#     if request.method == 'POST':
#         f = request.files['file']
#         basepath = os.path.dirname(__file__)
#         file_path = os.path.join(
#             basepath, 'uploads', secure_filename(f.filename))
#         f.save(file_path)
#         preds = model_predict(file_path, model)
#         result=preds
#         return result
#     return None


# if __name__ == '__main__':
#     app.run(port=5001,debug=True)


# from __future__ import division, print_function
# import os
# import cv2
# import numpy as np
# from flask import Flask, render_template, Response
# from tensorflow.keras.models import load_model

# app = Flask(__name__)

# # Load model
# MODEL_PATH = 'model.h5'
# model = load_model(MODEL_PATH)

# # Global for webcam
# camera = cv2.VideoCapture(0)

# # Preprocessing functions
# def grayscale(img):
#     return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# def equalize(img):
#     return cv2.equalizeHist(img)

# def preprocessing(img):
#     img = grayscale(img)
#     img = equalize(img)
#     img = img / 255
#     return img

# def getClassName(classNo):
#     class_names = [
#         'Speed Limit 20 km/h', 'Speed Limit 30 km/h', 'Speed Limit 50 km/h',
#         'Speed Limit 60 km/h', 'Speed Limit 70 km/h', 'Speed Limit 80 km/h',
#         'End of Speed Limit 80 km/h', 'Speed Limit 100 km/h', 'Speed Limit 120 km/h',
#         'No passing', 'No passing for vehicles over 3.5 metric tons',
#         'Right-of-way at the next intersection', 'Priority road', 'Yield', 'Stop',
#         'No vehicles', 'Vehicles over 3.5 metric tons prohibited', 'No entry',
#         'General caution', 'Dangerous curve to the left', 'Dangerous curve to the right',
#         'Double curve', 'Bumpy road', 'Slippery road', 'Road narrows on the right',
#         'Road work', 'Traffic signals', 'Pedestrians', 'Children crossing',
#         'Bicycles crossing', 'Beware of ice/snow', 'Wild animals crossing',
#         'End of all speed and passing limits', 'Turn right ahead', 'Turn left ahead',
#         'Ahead only', 'Go straight or right', 'Go straight or left', 'Keep right',
#         'Keep left', 'Roundabout mandatory', 'End of no passing',
#         'End of no passing by vehicles over 3.5 metric tons'
#     ]
#     return class_names[classNo]

# def generate_frames():
#     while True:
#         success, frame = camera.read()
#         if not success:
#             break
#         else:
#             img = cv2.resize(frame, (32, 32))
#             img = preprocessing(img)
#             img = img.reshape(1, 32, 32, 1)
#             predictions = model.predict(img)
#             classIndex = np.argmax(predictions)
#             className = getClassName(classIndex)

#             # Overlay class name
#             cv2.putText(frame, className, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
#             # Encode frame
#             ret, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()
            
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# @app.route('/')
# def index():
#     return render_template('index.html')  # Live stream template

# @app.route('/video')
# def video():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# if __name__ == '__main__':
#     app.run(debug=True, port=5001)


from flask import Flask, render_template, request, Response, redirect
from werkzeug.utils import secure_filename
import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
model = load_model('model.h5')

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def equalize(img):
    return cv2.equalizeHist(img)

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img

def getClassName(classNo):
    class_dict = {
        0: 'Speed Limit 20 km/h', 1: 'Speed Limit 30 km/h', 2: 'Speed Limit 50 km/h',
        3: 'Speed Limit 60 km/h', 4: 'Speed Limit 70 km/h', 5: 'Speed Limit 80 km/h',
        6: 'End of Speed Limit 80 km/h', 7: 'Speed Limit 100 km/h', 8: 'Speed Limit 120 km/h',
        9: 'No passing', 10: 'No passing for vehicles over 3.5 tons',
        11: 'Right-of-way at the next intersection', 12: 'Priority road',
        13: 'Yield', 14: 'Stop', 15: 'No vehicles',
        16: 'Vehicles over 3.5 tons prohibited', 17: 'No entry',
        18: 'General caution', 19: 'Dangerous curve to the left',
        20: 'Dangerous curve to the right', 21: 'Double curve',
        22: 'Bumpy road', 23: 'Slippery road', 24: 'Road narrows on the right',
        25: 'Road work', 26: 'Traffic signals', 27: 'Pedestrians',
        28: 'Children crossing', 29: 'Bicycles crossing', 30: 'Beware of ice/snow',
        31: 'Wild animals crossing', 32: 'End of all speed and passing limits',
        33: 'Turn right ahead', 34: 'Turn left ahead', 35: 'Ahead only',
        36: 'Go straight or right', 37: 'Go straight or left',
        38: 'Keep right', 39: 'Keep left', 40: 'Roundabout mandatory',
        41: 'End of no passing', 42: 'End of no passing for vehicles over 3.5 tons'
    }
    return class_dict.get(classNo, "Unknown")

def model_predict(img_path):
    img = image.load_img(img_path, target_size=(32, 32))
    img = np.asarray(img)
    img = preprocessing(img)
    img = img.reshape(1, 32, 32, 1)
    pred = model.predict(img)
    class_index = np.argmax(pred)
    return getClassName(class_index)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home', methods=['GET'])
def home():
    return render_template('/home.html')

@app.route('/about', methods=['GET'])
def about():
    return render_template('/about.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', result="No file part")
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', result="No file selected")
    if file:
        filepath = os.path.join('uploads', secure_filename(file.filename))
        file.save(filepath)
        result = model_predict(filepath)
        return render_template('index.html', result=result)

# ========== Live Video ==========
camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            img = cv2.resize(frame, (32, 32))
            img_proc = preprocessing(img)
            img_proc = img_proc.reshape(1, 32, 32, 1)
            pred = model.predict(img_proc)
            class_index = np.argmax(pred)
            label = getClassName(class_index)

            # Draw label on frame
            cv2.putText(frame, str(label), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 255, 0), 2, cv2.LINE_AA)

            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# @app.route('/video')
# def video():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video')
def video():
    return render_template("video.html")

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


#  Optional stop route (if needed)
@app.route('/stop')
def stop():
    
    return redirect('/')

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)


