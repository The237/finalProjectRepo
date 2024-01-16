from flask import Flask

app = Flask(__name__)

# Vous pouvez ajouter d'autres configurations ici si nécessaire
app.config["UPLOAD_FOLDER"] = "/chemin/vers/le/dossier/de/sauvegarde"

# Importez vos routes après avoir créé l'objet app
from app import routes
