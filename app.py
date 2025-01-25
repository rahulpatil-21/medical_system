from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import pickle
import numpy as np
import cv2
import sqlite3
import os
import secrets
from werkzeug.security import generate_password_hash, check_password_hash

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)

# SQLite database path
DATABASE = 'users.db'

# Function to get a database connection
def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row  # Allow accessing columns by name
    return conn

# Function to create users table if it doesn't exist
def init_db():
    with app.app_context():
        conn = get_db()
        conn.execute('''CREATE TABLE IF NOT EXISTS users (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            username TEXT UNIQUE NOT NULL,
                            password TEXT NOT NULL
                        )''')
        conn.commit()

# Initialize the database
init_db()

@app.route('/main_menu')
def main_menu():
    return render_template("main_menu.html")

@app.route('/', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Check if the username already exists
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        existing_user = cursor.fetchone()

        if existing_user:
            return "<h1>Username already exists. Please choose another.</h1>"

        # Hash the password using a supported method
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

        # Insert new user into the database
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
        conn.commit()

        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()

        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            return redirect(url_for('main_menu'))
        else:
            return "<h1>Invalid username or password</h1>"

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('login'))

@app.route('/anemia', methods=['GET', 'POST'])
def anemia():
   # if 'user_id' not in session:
    #    return redirect(url_for('login'))

    scaler = pickle.load(open('a1sc.pkl', 'rb'))
    classifier = pickle.load(open('anemiamodel.pkl', 'rb'))

    if request.method == "POST":
        try:
            gender = float(request.form['Gender'])
            hb = float(request.form['Hemoglobin'])
            mch = float(request.form['MCH'])
            mchc = float(request.form['MCHC'])
            mcv = float(request.form['MCV'])

            input_data = (gender, hb, mch, mchc, mcv)
            input_data_as_numpy_array = np.asarray(input_data)
            input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
            std_data = scaler.transform(input_data_reshaped)

            prediction = classifier.predict(std_data)
    
            if(prediction[0] == 1 ):
                result = "You have anemia" 
                color="red"
            else:
                result = "You do not have anemia"
                color="green"



            return render_template("res.html", result=result, color=color)
        except Exception as e:
            return render_template("res.html", result=f"Error: {str(e)}", color="red")

    return render_template("an.html")

@app.route('/brain', methods=["POST", "GET"])
def brain():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    with open('brainmodel.pkl', 'rb') as file:
        model = pickle.load(file)

    if request.method == "POST":
        try:
            if 'image' not in request.files:
                return jsonify({"Error": "No file part in the request"}), 400

            file = request.files['image']
            file.save("temp.jpg")

            c = cv2.imread("temp.jpg")
            k = cv2.resize(c, (256, 256))
            input_image = k.reshape((1, 256, 256, 3))
            prediction = model.predict(input_image)
            final = abs(prediction[0][0] - 1)
            if final == 1:
                result = "You have Brain Tumor"
                color = "red"
            else:
                result = "You do not have Brain Tumor"
                color = "green"

            return render_template("res.html", result=result, color=color)

        except Exception as e:
            return render_template("res.html", result=f"Error: {str(e)}", color="red")

    return render_template("image.html")

@app.route('/diabetes', methods=["POST", "GET"])
def diabetes():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    with open('diabscaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    model = pickle.load(open('dm1.h5', 'rb'))

    if request.method == "POST":
        try:
            pre = float(request.form['pregnancies'])
            glu = float(request.form['glucose'])
            bp = float(request.form['bloodPressure'])
            st = float(request.form['skinThickness'])
            ins = float(request.form['insulin'])
            weight = float(request.form['weight'])
            height = float(request.form['height'])
            fun = 0.627
            age = float(request.form['age'])
            bmi = float(weight / (height * height))

            input_data = np.array([[pre, glu, bp, st, ins, bmi, fun, age]])
            prediction = model.predict(input_data)

            result = "Positive for diabetes" if prediction[0][0] > 0.5 else "Negative for diabetes"
            prediction_proba = prediction[0][0] * 100

            return f"""
                <h1>Prediction: {result}</h1>
                <h2>Probability: {prediction_proba:.2f}%</h2>
            """
        except Exception as e:
            return jsonify({"Error": str(e)})

    return render_template("diabindex.html")

if __name__ == '__main__':
    app.run(debug=True)
