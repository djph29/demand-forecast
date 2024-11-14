# app.py
from flask import Flask, request, jsonify
import pandas as pd
import lightgbm as lgb
from train.train_pipeline import SalesForecastPipeline  # Assuming train class in `train` dir

app = Flask(__name__)

# Initialize and train the model on container start
pipeline = SalesForecastPipeline(train_data_path='./data/train.csv', test_data_path='./data/test.csv')
pipeline.run()  # This will load data, train the model, and save it

# Load the trained model
model = lgb.Booster(model_file='./model/lightgbm_model.txt')

from flask import Flask, request, jsonify
from datetime import datetime
import lightgbm as lgb

app = Flask(__name__)

# Load your trained model here
model = lgb.Booster(model_file='model/lightgbm_model.txt')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Ensure the input has required fields
    if 'date' not in data or 'store' not in data or 'item' not in data:
        return jsonify({"error": "Invalid input format"}), 400

    # Extract date and other fields
    try:
        date = datetime.strptime(data['date'], '%Y-%m-%d')
        month = data.get('month', date.month)  # Use provided month or extract from date
        day = data.get('day', date.weekday())  # Use provided day or extract from date
        year = data.get('year', date.year)     # Use provided year or extract from date
    except ValueError:
        return jsonify({"error": "Invalid date format. Use YYYY-MM-DD."}), 400

    # Prepare input for prediction
    input_data = [[data['store'], data['item'], month, day, year]]
    
    # Perform prediction
    prediction = model.predict(input_data)
    
    return jsonify({"predictions": prediction.tolist()})


@app.route('/status', methods=['GET'])
def status():
    return jsonify({'message': 'API is running'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)

