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


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse request JSON
        data = request.get_json()
        df = pd.DataFrame([data])  # Convert input to DataFrame

        # Ensure data has the correct features for prediction
        if all(col in df.columns for col in ['store', 'item', 'month', 'day', 'year']):
            predictions = model.predict(df[['store', 'item', 'month', 'day', 'year']])
            return jsonify({'predictions': predictions.tolist()})
        else:
            return jsonify({'error': 'Invalid input format'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/status', methods=['GET'])
def status():
    return jsonify({'message': 'API is running'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)

