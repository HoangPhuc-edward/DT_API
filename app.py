from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import ipaddress

# Load model
model = joblib.load("tree_model_depth_3.joblib")

# Init Flask app
app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        df = pd.read_csv(file)

        df = df.drop(columns=['Timestamp', 'Flow ID'], errors='ignore')
        df['Source IP'] = df['Source IP'].apply(lambda x: int(ipaddress.IPv4Address(x)))
        df['Destination IP'] = df['Destination IP'].apply(lambda x: int(ipaddress.IPv4Address(x)))
        df.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
        df = df.dropna()

        prediction = model.predict(df)
        return jsonify({'predictions': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
