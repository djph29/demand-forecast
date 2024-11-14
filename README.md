Demand Forecasting API
This project provides a demand forecasting model designed to predict item sales based on historical data. Two versions of the model are available: one using LightGBM and another using a TensorFlow neural network. A Flask-based REST API is used to serve predictions and check API status.
The tensorflow code is provided just as an alternative. The lightgmb model has been containerized for deployment.
Table of Contents
	•	Project Overview
	•	Features
	•	Project Structure
	•	Setup
	•	Usage
	•	API Endpoints
	•	Docker Deployment
	•	Dockerfile Explanation
	•	Model Training Explanations

Project Overview
The demand forecasting model predicts sales for specific dates, stores, and items, leveraging data preprocessing and training pipelines. 
Features
	•	Model Training: Two model options are available for training:
	◦	LightGBM: A gradient boosting model optimized for performance.
	◦	TensorFlow: A neural network regression model suitable for deep learning tasks.
	•	Prediction API: A REST API for generating predictions based on input data.
	•	Health Check Endpoint: An endpoint to check the API service status.
Project Structure


.
├── api/
│   ├── app.py              # Flask API server
├── train/
│   ├── train_pipeline.py    # Data preprocessing and LightGBM model training pipeline
│   ├── train_tf.py          # Data preprocessing and TensorFlow model training pipeline
├── model/
│   └── lightgbm_model.txt   # Saved LightGBM model (generated after training)
├── data/
│   ├── train.csv            # Training data
│   ├── test.csv             # Test data
├── Dockerfile               # Docker configuration for building the API service
└── requirements.txt         # Python dependencies
Setup
Prerequisites
	•	Python 3.9 or later
	•	Install required packages:  pip install -r requirements.txt  
Training the Model
LightGBM Version
To train the LightGBM model, run:
bash

python train/train_pipeline.py 
TensorFlow Version
To train the TensorFlow model, run:
bash

python train/train_tf.py
These scripts load training data, preprocess it, train the model, and save it in the model/ directory.
Usage
Running the API
To start the Flask API server locally, run:

python api/app.py
The server will start and listen on http://127.0.0.1:5001 by default.
API Endpoints
1. /predict
	•	Method: POST
	•	Description: Predicts the number of items sold based on input date, store, and item ID.
	•	Request Body:json   {
	•	  "date": "2013-01-01",
	•	  "store": 1,
	•	  "item": 1
	•	}
	•	  
	 
2. /status
	•	Method: GET
	•	Description: Returns the API status code to verify that the service is running.
	•	Response:json  
Docker Deployment
You can deploy the API as a Docker container.
Building the Docker Image
Checkout code:
git clone -b master https://github.com/djph29/demand-forecast.git
cd demand-forecast
Copy train.csv and test.csv to the data folder

In the project root, run:

docker build -t usfoods-api .
Running the Docker Container
To run the Docker container, use:
bash

docker run -p 5001:5001 usfoods-api
If 5001 is used by another application, it may have to be changed.
Accessing the API
After running the container, the API will be accessible at http://localhost:5001.
Dockerfile Explanation
Below is the Dockerfile configuration:
dockerfile

# Use a slim Python 3.9 image
FROM python:3.9-slim

# Install necessary dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files
COPY . /app

# Expose port for the API
EXPOSE 5001

# Set the default command to run the Flask API
CMD ["python", "api/app.py"]
Explanation of Key Dockerfile Instructions
	1	Base Image: Uses python:3.9-slim as a lightweight Python environment.
	2	Install Dependencies: Installs system-level dependencies (gcc and libgomp1) required by LightGBM.
	3	Working Directory: Sets the /app directory for storing application files.
	4	Install Python Packages: Installs Python dependencies from requirements.txt.
	5	Expose Port: Opens port 5001 for accessing the API.
	6	Command: Runs the Flask server when the container starts.

Model Training Explanations
LightGBM Model
LightGBM is a gradient boosting framework optimized for performance and efficiency. In this project, the LightGBM model is configured with the following parameters:
	•	objective: Regression with L1 loss for forecasting continuous values.
	•	metric: MAPE (Mean Absolute Percentage Error) for tracking model accuracy during training.
	•	learning_rate: Controls the rate of model learning.
	•	num_boost_round: Sets the number of boosting rounds.
The training script:
	1	Loads and preprocesses the data, splitting it into training and validation sets.
	2	Initializes and trains the LightGBM model.
	3	Saves the trained model as lightgbm_model.txt.
TensorFlow Model
The TensorFlow model is a neural network regression model that predicts sales based on the input data. The model consists of dense (fully connected) layers to capture patterns in the data.
train_tf.py Explanation
The train_tf.py script implements a TensorFlow-based neural network regression model. It performs the following steps:
	1	Data Preprocessing: Converts the date column to month, day, and year features. Splits the data into training and testing sets.
	2	Model Definition:
	◦	Defines a sequential model with dense layers.
	◦	The input layer consists of features (store, item, month, day, year).
	◦	Hidden layers are dense layers with ReLU activations to capture complex patterns.
	◦	The output layer has a single neuron to predict sales.
	3	Compilation:
	◦	Uses Mean Squared Error (MSE) as the loss function for regression.
	◦	Adam optimizer is used to minimize the loss function.
	4	Model Training:
	◦	The model is trained for a specified number of epochs, with validation on a held-out test set

This README provides a complete overview of the demand forecasting project, including the setup, usage, Docker deployment, and model training for both LightGBM and TensorFlow implementations.
Sample output:

deshane@deshane-HP-Laptop-17-by4xxx:~$ curl -X POST -H "Content-Type: application/json" -d '{
  "date": "2013-01-01",
  "store": 1,
  "item": 1
}' http://127.0.0.1:5001/predict
{"predictions":[9.534609697136744]}
deshane@deshane-HP-Laptop-17-by4xxx:~$ curl http://localhost:5001/status
{"message":"API is running"}

4o







A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A
A

