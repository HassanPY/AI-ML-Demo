'''
Building a simple Keras + deep learning REST API
Mon 29 January 2018
By Adrian Rosebrock
'''
# on windows install pip and curl

# We will present a simple method to take a Keras model and deploy it as a REST API.
# pip install keras
# pip install tensorflow , if needed


# import the necessary packages
from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import flask
import io

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None

'''
Our first code snippet handles importing our required packages and initializing 
both the Flask application and our model.
From there we define the load_model function:
'''

def load_model():
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras)
    global model
    model = ResNet50(weights="imagenet")
'''
As the name suggests, this method is responsible for instantiating our architecture and
loading our weights from disk.

For the sake of simplicity, we'll be utilizing the ResNet50 architecture 
which has been pre-trained on the ImageNet dataset.

If you're using your own custom model you'll want to modify this function to load your 
architecture + weights from disk.

Before we can perform prediction on any data coming from our client 
we first need to prepare and preprocess the data:
'''

def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    # return the processed image
    return image
'''
This function:

Accepts an input image
Converts the mode to RGB (if necessary)
Resizes it to 224x224 pixels (the input spatial dimensions for ResNet)
Preprocesses the array via mean subtraction and scaling
Again, you should modify this function based on any preprocessing, scaling,
and/or normalization you need prior to passing the input data through the model.

We are now ready to define the predict function — t
his method processes any requests to the /predict endpoint:
'''


@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            # preprocess the image and prepare it for classification
            image = prepare_image(image, target=(224, 224))

            # classify the input image and then initialize the list
            # of predictions to return to the client
            preds = model.predict(image)
            results = imagenet_utils.decode_predictions(preds)
            data["predictions"] = []

            # loop over the results and add them to the list of
            # returned predictions
            for (imagenetID, label, prob) in results[0]:
                r = {"label": label, "probability": float(prob)}
                data["predictions"].append(r)

            # indicate that the request was a success
            data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)

'''
The data dictionary is used to store any data that we want to return to the client. 
Right now this includes a boolean used to indicate if prediction was successful or not — 
we'll also use this dictionary to store the results of any predictions 
we make on the incoming data.

To accept the incoming data we check if:

The request method is POST (enabling us to send arbitrary data to the endpoint, i
ncluding images, JSON, encoded-data, etc.)
An image has been passed into the files attribute during the POST
We then take the incoming data and:

Read it in PIL format
Preprocess it
Pass it through our network
Loop over the results and add them individually to the data["predictions"] list
Return the response to the client in JSON format
If you're working with non-image data you should remove the request.files 
code and either parse the raw input data yourself or utilize request.get_json() to 
automatically parse the input data to a Python dictionary/object. 

All that's left to do now is launch our service:
'''
# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    load_model()
    app.run()

# From terminal run:
# python run_keras_server.py
# From another terminal run:
# curl -X POST -F image=@dog.jpg 'http://127.0.0.1:5000/predict' | python3 -m json.tool

# The -X flag and POST value indicates we're performing a POST request.

# We supply -F image=@dog.jpg to indicate we're submitting form encoded data. The image key is then set to the contents of the dog.jpg file. Supplying the @ prior to dog.jpg implies we would like cURL to load the contents of the image and pass the data to the request.

# endpoint: http://localhost:5000/predict

# python3 -m json.tool to Pretty JSON results