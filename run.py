from app import app

if __name__ == "__main__":
    print("Starting Python Flask Server for Cameroonian Stars Classification App !!!")
    # local config
    app.run(debug=True)
    # online config
    # app.run(host="0.0.0.0", port=8080)
