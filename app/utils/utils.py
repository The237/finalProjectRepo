import os
import numpy as np
import base64
import joblib
import cv2
import json
import matplotlib
from matplotlib import pyplot as plt
import pywt

__face_cascade = None
__eye_cascade = None
__path = None


# this method will just load all the variables and keep it as global for later use
def initialize_variables():
    print("We're in utils \n loading saved model ... start")
    global __face_cascade
    global __eye_cascade
    global __path

    __face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    __eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    # set the path
    __path = os.getcwd() + "\\models\\"


initialize_variables()


# classify image will get the cropped faces from the image and then convert it to wavelet,
# resize it and make a hot a vector from it to pass to the model
def classify(img_base64, model_name, img_path=None):
    # load the corresponding classes for a given model
    with open(__path + model_name + ".json", "r") as f:
        class_name_to_number = json.load(f)
        class_number_to_name = {k: v for v, k in class_name_to_number.items()}

    # load the corresponding classes for a given deep learning model
    with open(__path + model_name + "_dl.json", "r") as f:
        class_name_to_number_dl = json.load(f)
        class_number_to_name_dl = {k: v for v, k in class_name_to_number_dl.items()}

    # load the good model
    model = None
    if model is None:
        with open(
            __path + "machine_learning" + "\\svm_" + model_name + ".pkl", "rb"
        ) as f:
            model = joblib.load(f)
        print("Loading saved model done !!!")

    # load good model for deep learning
    # model_dl = None
    # model_dl = load_model(
    #     __path + "deep_learning\\" + model_name + "best_model" + ".h5"
    # )
    # print("Loading deep learning model done !!!")

    cropped_faces = get_cropped_image_if_2_eyes_exist(img_path, img_base64)
    result = []
    result_dl = []

    for face in cropped_faces:
        scalled_img = cv2.resize(face, (64, 64))
        img_wav = w2d(face, "db1", 5)
        scalled_img_wav = cv2.resize(img_wav, (64, 64))
        combined_img = np.vstack(
            (scalled_img.reshape(64 * 64 * 3, 1), scalled_img_wav.reshape(64 * 64, 1))
        )
        len_img_array = 64 * 64 * 3 + 64 * 64
        input = combined_img.reshape(1, len_img_array).astype(float)

        model_result = model.predict(input)

        result.append(
            {
                "class": class_number_to_name[model_result[0]],
                "class_probability": np.around(
                    model.predict_proba(input) * 100, 2
                ).tolist()[0],
                "class_dictionary": class_name_to_number,
            }
        )
        # nn_face = preprocess_image_for_neural_network(face)
        # dl_pred = model_dl.predict(nn_face)

        # result_dl.append(
        #     {
        #         "class": process_neural_network_result(
        #             dl_pred, class_number_to_name_dl
        #         )["class_neural_network"],
        #         "class_probability": np.around(
        #             model_dl.predict(nn_face) * 100, 2
        #         ).tolist()[0],
        #         "class_dictionary": class_name_to_number_dl,
        #     }
        # )
    # result_combined = {"svm": result, "dl": result_dl}
    result_combined = {"svm": result}
    return result_combined


# just a simple utitly function to ocnvert the class number to class name(string)
def class_number_to_name(class_num):
    return class_name_to_number[class_num]


# get the cropped faces from an image
def get_cropped_image_if_2_eyes_exist(img_path, image_base64):
    if img_path:
        img = cv2.imread(img_path)
    else:
        img = get_cv2_image_from_base64_string(image_base64)
    if img is None:
        return
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = __face_cascade.detectMultiScale(img_gray, 1.3, 5)
    cropped_faces = []
    for x, y, w, h in faces:
        roi_gray = img_gray[y : y + h, x : x + w]
        roi_color = img[y : y + h, x : x + w]
        eyes = __eye_cascade.detectMultiScale(roi_gray, 1.1, 5)
        if len(eyes) >= 2:
            cropped_faces.append(roi_color)
    return cropped_faces


# convert a string image base64 to a cv2 image
def get_cv2_image_from_base64_string(b64str):
    encoded_data = b64str.split(",")[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def w2d(img, mode="haar", level=1):
    imArray = img
    # Datatype conversions
    # convert to grayscale
    imArray = cv2.cvtColor(imArray, cv2.COLOR_RGB2GRAY)
    # convert to float
    imArray = np.float32(imArray)
    imArray /= 255
    # compute coefficients
    coeffs = pywt.wavedec2(imArray, mode, level=level)

    # Process Coefficients
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0

    # reconstruction
    imArray_H = pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255
    imArray_H = np.uint8(imArray_H)

    return imArray_H


def preprocess_image_for_neural_network(img_array):
    # Convertir l'array en image PIL
    img = image.array_to_img(img_array)

    # Redimensionner l'image à la taille attendue par votre modèle
    img = img.resize(
        (224, 224)
    )  # ajustez la taille en fonction des besoins de votre modèle

    # Convertir l'image PIL en tableau NumPy
    img_array = image.img_to_array(img)

    # Ajouter une dimension supplémentaire pour créer un lot (batch)
    img_array = np.expand_dims(img_array, axis=0)

    # Prétraiter l'image pour correspondre aux prétraitements utilisés lors de l'apprentissage
    img_array = preprocess_input(img_array)

    return img_array


def predict_neural_network(img_array, neural_network_model):
    # Prétraitez l'image si nécessaire (ajustez en fonction de vos besoins)
    preprocessed_img = preprocess_image_for_neural_network(img_array)

    # Faites la prédiction avec le modèle de réseau de neurones
    neural_network_result = neural_network_model.predict(preprocessed_img)

    # Retournez les résultats (ajustez en fonction de vos besoins)
    return neural_network_result


def process_neural_network_result(neural_network_result, class_number_to_name_dl):
    class_index = np.argmax(neural_network_result)
    class_name = class_number_to_name_dl[class_index]
    probability = neural_network_result[0][class_index]
    return {
        "class_neural_network": class_name,
        "class_probability_neural_network": probability,
    }
