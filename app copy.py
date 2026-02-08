from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load('screen_time_model.joblib')

def format_time(minutes):
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    return f"{hours} hours {mins} minutes"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    stress = int(request.form['stress'])
    happiness = int(request.form['happiness'])
    age = int(request.form['age'])

    input_frame = pd.DataFrame({'perceived_stress_score': [stress], 'self_reported_happiness': [happiness], 'age': [age]})
    prediction = model.predict(input_frame)

    return render_template('result.html', low=format_time(prediction[0] - 30), high=format_time(prediction[0] + 30))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
