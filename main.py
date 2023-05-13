from flask import Flask, request, render_template
import base64
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
import cv2

np.set_printoptions(suppress=True)

# Load the model
model = load_model("model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1
def process_img(img_name):
  img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
  if (img is None):
    print('read unsuccess')
  print(img.shape)
  img = np.array(cv2.resize(img, (256, 256)))
  img = img.reshape(1,256,256,1)
  #img = img//255
  return img

app = Flask(__name__)
"""
	        image = Image.open(img_data.filename).convert("RGB")
	        size = (224, 224)
	        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
	        image_array = np.asarray(image)
	        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
	        data[0] = normalized_image_array
	        prediction = model.predict(data)
	        index = np.argmax(prediction)
	        class_name = class_names[index]
	        confidence_score = prediction[0][index]
	        message = class_name
	        """
@app.route('/', methods=['GET'])
def index():
    if request.method == 'GET':
        return render_template('index_old.html')

@app.route('/ai', methods=['POST'])
def ai_python():
    if request.method == 'POST':
        # รับภาพที่ส่งมาจาก HTML form
        print('list',request.files,request.form['image'])
        image_data = request.form['image']
        image_binary = base64.b64decode(image_data.split(',')[1])
        with open('a.jpg', 'wb') as file:
            file.write(image_binary)
            
        image = Image.open('a.jpg').convert("RGB")
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data = process_img('a.jpg')
        prediction = model.predict(data)
        index = np.argmax(prediction)            
        class_name = ['ก','ข','ต','ธ','บ','ภ','ว','ห'][index]
        confidence_score = prediction[0][index]
        message = class_name
        return message
    else:
        print('no')
        return "none"

if __name__ == '__main__':
    app.run(debug=True)
