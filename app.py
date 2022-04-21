import tensorflow as tf
import numpy as np
import os
import urllib.request
from tensorflow.keras.preprocessing import image
from flask import Flask, request,  jsonify
from werkzeug.utils import secure_filename

upload_folder = "static/container"

app = Flask(__name__,
            template_folder = "../templates",
            static_folder = "../static")
app.secret_key = "secret key"
app.config["UPLOAD_FOLDER"] = upload_folder
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

extensions = set(["png", "jpg", "jpeg"])

def extension(filename):
	return "." in filename and filename.rsplit(".", 1)[1].lower() in extensions

def model(filename):
    model = tf.keras.models.load_model("model_sota.h5")
    src_img = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    img = image.load_img(src_img, target_size=(480, 640))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = np.vstack([img])
    result = model.predict(img)
    return result
	
@app.route("/v1/")
def return_home_message():
	data = {
		"status": 200,
		"message": "Hello to the computer vision prediction"
	}
	return jsonify(data)

@app.route("/v1/predict", methods=["POST"])
def predict():
	if "file" not in request.files:
		data = {
			"status": 400,
			"message": "File does not exist"
		}
		return jsonify(data)
	file = request.files['file']
	
	# If the user does not select a file, then return 400 with message
	if file.filename == "":
		data = {
			"status": 400,
			"message": "No image selected for uploading"
		}
		return jsonify(data)

	#if the user upload any other files than png, jpg, jpeg
	if extension(file.filename) == False:
		data = {
			"status": 400,
			"message": "incorrect image format for uploading"
		}
		return jsonify(data)
	# print(secure_filename(file.filename))
	filename = secure_filename(file.filename)
	file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
	pred = model(filename)
	data = {
		"status": 200,
		"number_people":  int(round(pred[0,0])),
		"percentage_people": int(round(pred[0,0])) / 84
	}
	return jsonify(data)
	

# @app.route('/display/<filename>')
# def display_image(filename):
# 	#print('display_image filename: ' + filename)
# 	return redirect(url_for("static", filename="container/"+filename), code=301)
# else:
	# 	data = {
	# 		"status": 400,
	# 		"message": "Allowed image types are : png, jpg, jpeg"
	# 	}
	# 	return jsonify(data)
	# if file and extension(file.filename):
	# 	

if __name__ == "__main__":
    app.run(debug=True)
