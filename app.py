from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained models
with open('LR_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

with open('standard_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')  # HTML form for input fields

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Extract form data
    country = request.form['country']
    gender = request.form['gender']
    age = int(request.form['age'])
    products_number = int(request.form['products_number'])
    tenure = int(request.form['tenure'])
    balance = float(request.form['balance'])
    credit_card = request.form['credit_card']
    active_member = request.form['active_member']
    estimated_salary = float(request.form['estimated_salary'])

    # Map categorical inputs to numeric
    country_map = {"France": 0, "Spain": 2, "Germany": 1}
    country_numeric = country_map[country]

    gender_map = {"Female": 0, "Male": 1}
    gender_numeric = gender_map[gender]

    credit_map = {"No": 0, "Yes": 1}
    credit_numeric = credit_map[credit_card]

    active_map = {"No": 0, "Yes": 1}
    active_numeric = active_map[active_member]

    # Create DataFrame from user input
    input_data = pd.DataFrame({
        'credit_score': [np.random.randint(300, 850)],  # Example random credit score
        'country': [country_numeric],
        'gender': [gender_numeric],
        'age': [age],
        'tenure': [tenure],
        'balance': [balance],
        'products_number': [products_number],
        'credit_card': [credit_numeric],
        'active_member': [active_numeric],
        'estimated_salary': [estimated_salary]
    })

    # Preprocess input data
    input_data_scaled = scaler.transform(input_data)

    # Make prediction
    churn_prediction = rf_model.predict(input_data_scaled)

    # Create result message
    if churn_prediction[0] == 1:
        result = "Customer is likely to leave the Bank."
        color = "red"
    else:
        result = "Customer is unlikely to leave the Bank."
        color = "green"

    # Return rendered HTML with the result
    return render_template('result.html', result=result, color=color)

if __name__ == '__main__':
    app.run(debug=True)
