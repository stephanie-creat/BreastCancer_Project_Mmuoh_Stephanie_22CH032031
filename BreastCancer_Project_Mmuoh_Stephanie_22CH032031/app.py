from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('model/breast_cancer_model.pkl')
scaler = joblib.load('model/scaler.pkl')

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_text = ""

    if request.method == 'POST':
        features = [
            float(request.form['radius']),
            float(request.form['texture']),
            float(request.form['perimeter']),
            float(request.form['area']),
            float(request.form['concavity'])
        ]

        features = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)

        if prediction[0] == 1:
            prediction_text = "Malignant Tumor"
        else:
            prediction_text = "Benign Tumor"

    return render_template('index.html', prediction=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
