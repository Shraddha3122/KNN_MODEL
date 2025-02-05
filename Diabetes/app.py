# app.py
from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Load the dataset
data = pd.read_csv('D:/WebiSoftTech/KNN MODEL/Diabetes/diabetes.csv')

# Preprocess the data
X = data.drop('Outcome', axis=1)  # Features
y = data['Outcome']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train the KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('D:/WebiSoftTech/KNN MODEL/Diabetes/index.html')

from flask import jsonify

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    features = [float(x) for x in request.form.values()]
    features = scaler.transform([features])  # Standardize the input
    prediction = knn.predict(features)

    # Return a JSON response
    if prediction[0] == 1:
        return jsonify({"patient_type": "Diabetic"})
    else:
        return jsonify({"patient_type": "Non-Diabetic"})

if __name__ == "__main__":
    app.run(debug=True)