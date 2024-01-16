from flask import render_template, request, jsonify
from app import app
import os
from werkzeug.utils import secure_filename
from app.utils.utils import classify
from app.modules import classification_models


@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")


@app.route("/football", methods=["GET", "POST"])
def football():
    return render_template("football.html", page="football")


@app.route("/athletisme", methods=["GET", "POST"])
def athletisme():
    return render_template("athletisme.html", page="athletisme")


@app.route("/sport_combat", methods=["GET", "POST"])
def sport_combat():
    return render_template("sport_combat.html", page="sport_combat")


@app.route("/education", methods=["GET", "POST"])
def education():
    return render_template("education.html", page="education")


@app.route("/cinema", methods=["GET", "POST"])
def cinema():
    return render_template("cinema.html", page="cinema")


@app.route("/informatique", methods=["GET", "POST"])
def informatique():
    return render_template("informatique.html", page="informatique")


@app.route("/fashion", methods=["GET", "POST"])
def fashion():
    return render_template("fashion.html", page="fashion")


@app.route("/classify_image", methods=["POST"])
def classify_image():
    image_data = request.form["image_data"]
    model_name = request.form["model_name"]

    prediction = jsonify(classify(image_data, model_name))

    return prediction
