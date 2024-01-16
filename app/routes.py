from flask import render_template, request, jsonify, send_file
from app import app
import os
from werkzeug.utils import secure_filename
from app.utils.utils import classify
from app.modules import classification_models
import pandas as pd

# import for graphics
import matplotlib.pyplot as plt
from io import BytesIO
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


# route pour la page home
@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")


# route pour la page football
@app.route("/football", methods=["GET", "POST"])
def football():
    return render_template("football.html", page="football")


# route pour la page athlétisme
@app.route("/athletisme", methods=["GET", "POST"])
def athletisme():
    return render_template("athletisme.html", page="athletisme")


# route pour la page sport de combats
@app.route("/sport_combat", methods=["GET", "POST"])
def sport_combat():
    return render_template("sport_combat.html", page="sport_combat")


# route pour la page éducation
@app.route("/education", methods=["GET", "POST"])
def education():
    return render_template("education.html", page="education")


# route pour la page cinéma
@app.route("/cinema", methods=["GET", "POST"])
def cinema():
    return render_template("cinema.html", page="cinema")


# route pour la page informatique
@app.route("/informatique", methods=["GET", "POST"])
def informatique():
    return render_template("informatique.html", page="informatique")


# route pour la page fashion
@app.route("/fashion", methods=["GET", "POST"])
def fashion():
    return render_template("fashion.html", page="fashion")


# route d'affichage des statistiques de classification
@app.route("/statistics")
def statistics():
    return render_template("statistics.html", page="statistics")


@app.route("/statistics_graph", methods=["GET"])
def statistics_graph():
    celebrity_data = {
        "AJARA": 48,
        "ONANA": 54,
        "MBANGO": 22,
        "DONGMO": 18,
        "ATANGANA": 52,
        "NTEFF": 35,
        "NDOUMBE": 24,
        "NGANNOU": 52,
        "BEYALA": 73,
        "AMADOU_AMAL": 45,
        "AYENA": 33,
        "EDIMA": 49,
        "KEPOMBIA": 10,
        "MEMBA": 15,
    }

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.barh(list(celebrity_data.keys()), list(celebrity_data.values()))
    plt.ylabel("Célébrité")
    plt.xlabel("Nombre d'images")
    # plt.xticks(rotation=-75)
    plt.title("Nombre d'images par célébrité")

    canvas = FigureCanvas(fig)
    img = BytesIO()
    fig.savefig(img)
    img.seek(0)

    # Renvoyer le fichier image en tant que réponse à la requête
    return send_file(img, mimetype="image/png")


# route de classification des images
@app.route("/classify_image", methods=["POST"])
def classify_image():
    image_data = request.form["image_data"]
    model_name = request.form["model_name"]

    prediction = jsonify(classify(image_data, model_name))

    return prediction
