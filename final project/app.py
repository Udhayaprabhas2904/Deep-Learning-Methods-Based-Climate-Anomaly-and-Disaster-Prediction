from flask import Flask, render_template, request, send_file, send_from_directory
import cv2
import pytesseract
import os
from werkzeug.utils import secure_filename
import numpy as np
import tensorflow as tf
import pandas as pd
from PIL import Image
import re
import io, random
from datetime import date
app = Flask(__name__)

model = tf.keras.models.load_model("dcnn_disaster_model_final.h5")
#df = pd.read_csv("Disaster.csv") 
df = pd.read_csv("Disaster_5000.csv") 
labels = [
    "flood", "cyclone_storm", "heatwave",
    "landslide", "wildfire", "volcanic_eruption"
]

# Set Tesseract path (your local path)
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\narma\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# Regex to find GPS coordinates like: 8.72251, 77.73891
coord_pattern = re.compile(r"(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)")


# Folder to save uploaded/processed images
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/login")
def login():
    return render_template("login.html")

@app.route("/logincheck", methods=["POST"])
def logincheck():
    email = request.form.get("email")
    password = request.form.get("password")

    if email == "narmatha@gmail.com" and password == "1234":
        return render_template("userinput.html")

    # Send SweetAlert trigger to template
    return render_template("login.html", error="Invalid Username or Password")


@app.route("/sensor")
def sensor():
    return render_template("userinput.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        values = [
            float(request.form.get("temp_min")),
            float(request.form.get("temp_max")),
            float(request.form.get("temp_avg")),
            float(request.form.get("rainfall")),
            float(request.form.get("humidity")),
            float(request.form.get("soil"))
        ]

        data = np.array(values).reshape(1, 6, 1)
        prediction = model.predict(data, verbose=0)[0]

        result = {
            labels[i]: "⚠️ Risk" if prediction[i] > 0.40 else "✅ Safe"
            for i in range(6)
        }

        return render_template("result.html", result=result, values=values)

    except Exception as e:
        return render_template("index.html", error=str(e))

@app.route("/search", methods=["GET", "POST"])
def search():
    df.columns = df.columns.str.strip().str.lower()  # ensure clean columns

    countries = sorted(df['country'].dropna().unique())
    states, cities, dates = [], [], []
    results = pd.DataFrame()

    selected_country = request.form.get("country")
    selected_state = request.form.get("state")
    selected_city = request.form.get("city")
    selected_date = request.form.get("date")

    if selected_country:
        states = sorted(df[df['country']==selected_country]['state'].dropna().unique())
    if selected_state:
        cities = sorted(df[(df['country']==selected_country) & (df['state']==selected_state)]['city'].dropna().unique())
    if selected_city:
        dates = sorted(df[
            (df['country']==selected_country) &
            (df['state']==selected_state) &
            (df['city']==selected_city)
        ]['date'].dropna().unique())

    if selected_date:
        results = df[
            (df['country']==selected_country) &
            (df['state']==selected_state) &
            (df['city']==selected_city) &
            (df['date']==selected_date)
        ]

    return render_template(
        "search.html",
        countries=countries, states=states, cities=cities, dates=dates,
        results=results,
        selected_country=selected_country,
        selected_state=selected_state,
        selected_city=selected_city,
        selected_date=selected_date
    )



@app.route("/dashboard")
def dashboard():
    df.columns = df.columns.str.lower().str.strip()

    # COUNTRY COUNT
    country_count = df["country"].value_counts().to_dict()

    # STATE COUNT
    state_count = df["state"].value_counts().to_dict() if "state" in df.columns else {}

    # CITY COUNT
    city_count = df["city"].value_counts().to_dict() if "city" in df.columns else {}

    # Average Temperature by Country
    avg_temp_by_country = df.groupby("country")["temp_avg"].mean().to_dict() if "temp_avg" in df.columns else {}

    # DISASTER COUNT (columns where positive => risk happened)
    disaster_columns = ["flood","cyclone_storm","heatwave","landslide","wildfire","volcanic_eruption"]
    disaster_count = {}

    for col in disaster_columns:
        if col in df.columns:
            disaster_count[col] = int((df[col] > 0).sum())
        else:
            disaster_count[col] = 0

    return render_template(
        "dashboard.html",
        country_count=country_count,
        state_count=state_count,
        city_count=city_count,
        avg_temp_by_country=avg_temp_by_country,
        disaster_count=disaster_count
    )



##@app.route("/satelite", methods=["GET", "POST"])
##def upload_image():
##    location = None
##    map_link = None
##    error = None
##
##    if request.method == "POST":
##        try:
##            file = request.files["image"]
##            img_bytes = file.read()
##            img = Image.open(io.BytesIO(img_bytes))
##
##            # Extract text using Tesseract OCR
##            extracted_text = pytesseract.image_to_string(img)
##
##            # Search for coordinates
##            match = coord_pattern.search(extracted_text)
##
##            if match:
##                lat, lon = match.groups()
##                location = (lat, lon)
##                map_link = f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"
##            else:
##                error = "❌ Could not detect GPS coordinates from the image"
##
##        except Exception as e:
##            error = f"Error: {e}"
##
##    return render_template("satelite.html", 
##                            location=location,
##                            map_link=map_link,
##                            error=error)
##


@app.route("/satelite", methods=["GET", "POST"])
def upload_image():
    location = None
    map_link = None
    error = None
    data = None

    if request.method == "POST":
        try:
            file = request.files["image"]
            img_bytes = file.read()
            img = Image.open(io.BytesIO(img_bytes))

            extracted_text = pytesseract.image_to_string(img)

            coord_pattern = re.compile(r'([-+]?\d+\.\d+)[,\s]+([-+]?\d+\.\d+)')
            match = coord_pattern.search(extracted_text)

            if match:
                lat, lon = match.groups()
                location = (lat, lon)
                map_link = f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"

                # 🔥 RANDOM DATA GENERATION
                temp_min = round(random.uniform(15, 25), 1)
                temp_max = round(random.uniform(30, 42), 1)

                data = {
                    "temp_min_c": temp_min,
                    "temp_max_c": temp_max,
                    "temp_avg_c": round((temp_min + temp_max) / 2, 1),
                    "rainfall_mm": round(random.uniform(0, 300), 1),
                    "humidity_percent": random.randint(30, 95),
                    "soil_moisture": random.randint(10, 90),

                    "flood": random.choice([0, 1]),
                    "cyclone_storm": random.choice([0, 1]),
                    "heatwave": random.choice([0, 1]),
                    "landslide": random.choice([0, 1]),
                    "wildfire": random.choice([0, 1]),
                    "volcanic_eruption": random.choice([0, 1]),
                }

            else:
                error = "❌ Could not detect GPS coordinates from the image"

        except Exception as e:
            error = f"Error: {e}"

    return render_template(
        "satelite.html",
        location=location,
        map_link=map_link,
        error=error,
        date=date.today().strftime("%d-%m-%Y"),
        data=data
    )


@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)



