##Flight Price Pred App

import json
import boto3
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load model from S3 on startup
def load_model():
    s3 = boto3.client('s3')
    s3.download_file('your-bucket-name', 'models/xgb_tuned.json', '/tmp/xgb_tuned.json')
    model = XGBRegressor()
    model.load_model('/tmp/xgb_tuned.json')
    return model

model = load_model()

# Load saved feature column order
FEATURE_COLS = ['Airline', 'Source', 'Destination', 'Total_Stops',
                'Additional_Info', 'Journey_Day', 'Journey_Month',
                'Arrival_Time_Hour', 'Arrival_Time_Minute',
                'Dep_Time_Hour', 'Dep_Time_Minute',
                'Duration_hour', 'Duration_minutes', 'Duration_total_mins']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data   = request.get_json()
        df     = pd.DataFrame([data])
        df     = df[FEATURE_COLS]         # enforce column order
        price  = model.predict(df)[0]
        mape   = 12.79
        lower  = round(price * (1 - mape/100), 0)
        upper  = round(price * (1 + mape/100), 0)

        return jsonify({
            'predicted_price': round(float(price), 2),
            'price_range':     {'lower': lower, 'upper': upper},
            'confidence':      f'{100 - mape:.1f}%',
            'currency':        'INR'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model': 'xgb_tuned_v1'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
```

**Step 3 — `requirements.txt`**
```
flask==3.0.0
xgboost==2.0.3
pandas==2.1.0
numpy==1.26.0
boto3==1.34.0
scikit-learn==1.4.0