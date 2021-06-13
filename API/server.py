import cv2
import numpy as np
import pandas as pd
import tensorflow
from flask                                  import Flask, request, jsonify
from tensorflow.keras.models                import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

app = Flask(__name__)
model = load_model('model.h5')
@app.route('/api', methods = ['POST'])

def predict():    
    nparr            = np.fromstring(request.data, np.uint8) # convert string of image data to uint8
    img              = cv2.imdecode(nparr, cv2.IMREAD_COLOR)# decode image
    img              = cv2.resize(img, (224, 224), interpolation = cv2.INTER_AREA)
    img_batch        = np.expand_dims(img, axis = 0)
    img_preprocessed = preprocess_input(img_batch)
    prediction       = model.predict(img_preprocessed)
    proba            = str(round(prediction.max(axis = 1)[0], 3))
    classe           = pd.DataFrame(prediction).rename(columns = {0:'Cat', 1:'Dog'}).idxmax(axis = 1)[0]
    return jsonify([proba, classe])
    #return jsonify("hello world")

if __name__ == '__main__':
    app.run(debug = True, threaded=False)