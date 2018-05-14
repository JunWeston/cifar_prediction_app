import os
from flask import Flask, request, render_template, send_from_directory
from cifar10_web_api_eval import evaluate_with_api
import pandas as pd

FNM_SAVED_IMAGE = 'img_for_prediction.jpg'
TEMPLATES_FOLDER = 'templates'

app = Flask(__name__)
# app = Flask(__name__, static_folder="images")

APP_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))


@app.route("/")
def index():

    return render_template("upload.html")


@app.route("/upload", methods=["POST"])
def upload():
    
    target = os.path.join(APP_ROOT, 'images/')

    print(target)
    if not os.path.isdir(target):
            os.mkdir(target)
    else:
        print("Couldn't create upload directory: {}".format(target))
        
    # try to remove existing file
    try:
        os.remove(os.path.join(target,FNM_SAVED_IMAGE))
    except OSError:
        pass
    
    print(request.files.getlist("file"))
    for upload in request.files.getlist("file"):
        print(upload)
        print("{} is the file name".format(upload.filename))
        filename = upload.filename
        destination = "/".join([target, FNM_SAVED_IMAGE])
        print ("Accept incoming file:", filename)
        print ("Save it to:", destination)
        upload.save(destination)
        
    # obtain estimate from the CNN
    df = evaluate_with_api(image_fnm=FNM_SAVED_IMAGE)

    # render template with results
    return render_template("complete_show_results.html", 
                           table=df, 
                           image_name = FNM_SAVED_IMAGE)


@app.route('/upload/<filename>')
def send_image(filename):
    # return send_from_directory("images", filename)
    target = os.path.join(APP_ROOT, 'images/')
    return send_from_directory(target, FNM_SAVED_IMAGE)

if __name__ == "__main__":
    app.run(port=4555, debug=True)