# app.py

from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load the trained model
model_path = 'model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form
    int_features = [float(x) for x in request.form.values()]
    print("Received form data: ", int_features)

    final_features = np.array(int_features)
    final_features = final_features.reshape(1,-1)
    
    # Make prediction
    prediction = model.predict(final_features)
    output = prediction[0]

    return render_template('index.html', prediction_text='Prediction: {}'.format(output))

@app.route('/favicon.ico')
def favicon():
    return app.send_static_file('favicon.ico')


if __name__ == "__main__":
    app.run(debug=True)