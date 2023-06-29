import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from flask import Flask, flash, request, redirect, render_template
import os
from werkzeug.utils import secure_filename
import cv2
import imutils
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input


# Load pre-trained models
pneumonia_model = load_model('models/modelpneumonia.h5')
covid_model = load_model('models/modelcovid19.h5')
alzheimer_model = load_model('models/modelalzheimer.h5')
braintumor_model = load_model('models/modelbraintumor.h5')

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "secret key"

############################################# BRAIN TUMOR FUNCTIONS ################################################

def preprocess_imgs(set_name, img_size):
    """
    Resize and apply VGG-15 preprocessing to the images.
    """
    set_new = []
    for img in set_name:
        img = cv2.resize(img, dsize=img_size, interpolation=cv2.INTER_CUBIC)
        set_new.append(preprocess_input(img))
    return np.array(set_new)


def crop_imgs(set_name, add_pixels_value=0):
    """
    Find the extreme points on the image and crop a rectangular region based on them.
    """
    set_new = []
    for img in set_name:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Threshold the image and perform erosion and dilation operations to remove noise
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Find contours in the thresholded image and grab the largest one
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)

        # Find the extreme points
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])

        ADD_PIXELS = add_pixels_value
        new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS, extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()
        set_new.append(new_img)

    return np.array(set_new)

##################################################################################################################

def allowed_file(filename):
    """
    Check if the uploaded file has an allowed file extension.
    """
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('homepage.html')

@app.route('/covid')
def covid():
    return render_template('covid.html')

@app.route('/braintumor')
def brain_tumor():
    return render_template('braintumor.html')

@app.route('/alzheimer')
def alzheimer():
    return render_template('alzheimer.html')

@app.route('/pneumonia')
def pneumonia():
    return render_template('pneumonia.html')

###################################################################

@app.route('/resultc', methods=['POST'])
def resultc():
    if request.method == 'POST':
        file = request.files['file']

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('Image successfully uploaded and displayed below')
            img = cv2.imread('static/uploads/' + filename)
            img = cv2.resize(img, (224, 224))
            img = img.reshape(1, 224, 224, 3)
            img = img / 255.0
            pred = covid_model.predict(img)
            if pred < 0.5:
                pred = 0
            else:
                pred = 1

            return render_template('resultc.html', filename=filename, r=pred)
        else:
            flash('Allowed image types are - png, jpg, jpeg')
            return redirect(request.url)

@app.route('/resultbt', methods=['POST'])
def resultbt():
    if request.method == 'POST':
        file = request.files['file']

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('Image successfully uploaded and displayed below')
            img = cv2.imread('static/uploads/' + filename)
            img = crop_imgs([img])
            img = img.reshape(img.shape[1:])
            img = preprocess_imgs([img], (224, 224))
            pred = braintumor_model.predict(img)
            if pred < 0.5:
                pred = 0
            else:
                pred = 1

            return render_template('resultbt.html', filename=filename, r=pred)
        else:
            flash('Allowed image types are - png, jpg, jpeg')
            return redirect(request.url)

@app.route('/resulta', methods=['POST'])
def resulta():
    if request.method == 'POST':
        file = request.files['file']

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('Image successfully uploaded and displayed below')
            img = cv2.imread('static/uploads/' + filename)
            img = cv2.resize(img, (176, 176))
            img = img.reshape(1, 176, 176, 3)
            img = img / 255.0
            pred = alzheimer_model.predict(img)
            if pred[0][0] < 0.5:
                pred = 0
            else:
                pred = 1

            return render_template('resulta.html', filename=filename, r=pred)
        else:
            flash('Allowed image types are - png, jpg, jpeg')
            return redirect(request.url)

@app.route('/resultp', methods=['POST'])
def resultp():
    if request.method == 'POST':
        file = request.files['file']

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('Image successfully uploaded and displayed below')
            img = cv2.imread('static/uploads/' + filename)
            img = cv2.resize(img, (150, 150))
            img = img.reshape(1, 150, 150, 3)
            img = img / 255.0
            pred = pneumonia_model.predict(img)
            if pred < 0.5:
                pred = 0
            else:
                pred = 1

            return render_template('resultp.html', filename=filename, r=pred)
        else:
            flash('Allowed image types are - png, jpg, jpeg')
            return redirect(request.url)

# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response

if __name__ == '__main__':
    app.run(debug=True)
