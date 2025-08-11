from flask import Flask, render_template, request, redirect, url_for, session, flash 
import os
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import tensorflow as tf
import io
import base64
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.secret_key = 'your_secret_key'

basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'users.db')

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

mobilenet_model = tf.keras.models.load_model("models/mobilenet_model.h5")
class_names = ['Early Blight', 'Late Blight', 'Healthy']

users = {'admin': 'admin'}

@app.route('/')
def root():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = User.query.filter_by(username=username, password=password).first()
        if user:
            session['username'] = user.username
            session['role'] = user.role 
            flash('Login Successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid Credentials, Please try again.', 'danger')

    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        role = 'admin' if username == 'admin' else 'user'

        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Username already exists, Please choose another.', 'danger')
        else:
            new_user = User(username=username, password=password, role=role)
            db.session.add(new_user)
            db.session.commit()
            flash('Signup Successful! Please log in.', 'success')
            return redirect(url_for('login'))

    return render_template('signup.html')

@app.route('/home', methods=['GET', 'POST'])
def home():
    if 'username' not in session:
        return redirect(url_for('login'))

    image_url = None
    prediction = None

    if request.method == 'POST':
        image = request.files['crop_image']
        if image:
            img = Image.open(image)
            img_resized = img.resize((224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            preds = mobilenet_model.predict(img_array)
            predicted_class = class_names[np.argmax(preds)]
            prediction = predicted_class

            image_bytes = io.BytesIO()
            img.save(image_bytes, format='PNG')
            image_bytes.seek(0)
            image_url = "data:image/png;base64," + base64.b64encode(image_bytes.read()).decode('utf-8')

    return render_template('home.html', username=session['username'],
                           image_url=image_url,
                           prediction=prediction)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form['name']
        phone = request.form['phone']
        email = request.form['email']
        message = request.form['message']

        new_contact = Contact(name=name, phone=phone, email=email, message=message)
        db.session.add(new_contact)
        db.session.commit()
        session.clear()
        flash('Your message has been submitted successfully!', 'success')
        return redirect(url_for('contact'))

    return render_template('contact.html')

@app.route('/logout')
def logout():
    session.clear()
    session.pop('username', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/admin/dashboard')
def admin_dashboard():
    if 'role' not in session or session['role'] != 'admin':
        flash('Unauthorized access.', 'danger')
        return redirect(url_for('login'))

    users = User.query.all()
    messages = Contact.query.all()
    return render_template('admin_dashboard.html', users=users, messages=messages)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    role = db.Column(db.String(10), nullable=False, default='user')  # 'admin' or 'user'

class Contact(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    phone = db.Column(db.String(20), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    message = db.Column(db.Text, nullable=False)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)

