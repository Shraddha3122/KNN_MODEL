#Given the data of breast cancer patients. Find out whether the cancer is benign 
#or malignant with the help of K Nearest Neighbors Machine Learning model.



from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Load and preprocess the data
data = pd.read_csv('D:/WebiSoftTech/KNN MODEL/Breast Cancer/breast-cancer-wisconsin.data', header=None)

# Replace '?' with NaN
data.replace('?', np.nan, inplace=True)

# Convert columns to numeric, forcing errors to NaN
data = data.apply(pd.to_numeric, errors='coerce')

# Drop rows with NaN values
data.dropna(inplace=True)

# Assume the last column is the label and the rest are features
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Create Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from request
    data = request.get_json(force=True)
    
    # Extract features from the request
    # Adjust the number of features according to your dataset
    features = [
        data['feature1'], 
        data['feature2'], 
        data['feature3'], 
        data['feature4'], 
        data['feature5'], 
        data['feature6'], 
        data['feature7'], 
        data['feature8'], 
        data['feature9'], 
        data['feature10']  # Ensure this matches the number of features in your dataset
    ]

    # Scale the features
    features = scaler.transform([features])  # Scale the input features
    prediction = knn.predict(features)
    
    # Return the prediction
    return jsonify({'prediction': 'benign' if prediction[0] == 0 else 'malignant'})

if __name__ == '__main__':
    app.run(debug=True)