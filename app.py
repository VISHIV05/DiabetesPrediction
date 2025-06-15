import os
import pandas as pd
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import numpy as np
from datetime import datetime
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# File paths
MODEL_PATH = 'model.pkl'
REAL_TIME_PREDICTIONS_PATH = 'data/real_time_predictions.csv'
BATCH_PREDICTIONS_PATH = 'data/batch_predictions.csv'
ONLINE_DATA_PATH = 'data/online_data.csv'
MODEL_METRICS_PATH = 'data/model_metrics.json'

REQUIRED_FEATURES = ['Pregnancies', "Glucose", "BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]

def fetch_and_save_data():
    '''Fetch the dataset from an online API'''
    url = f'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
    columns = REQUIRED_FEATURES + ['Outcome']
    data = pd.read_csv(url, header=None, names = columns)
    os.makedirs("data", exist_ok=True)
    data.to_csv(ONLINE_DATA_PATH, index=False)
    print("Dataset downloaded and saved into data folder")
    return data

def train_and_save_model():
    '''Training the model and save it to a file'''
    if not os.path.exists(ONLINE_DATA_PATH):
        data = fetch_and_save_data()
    else:
        data = pd.read_csv(ONLINE_DATA_PATH)
    
    X = data.drop(columns = ['Outcome'])
    y = data['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Save model metrics
    metrics = {
        'accuracy': accuracy,
        'classification_report': report,
        'training_date': datetime.now().isoformat(),
        'feature_names': REQUIRED_FEATURES
    }
    
    os.makedirs("data", exist_ok=True)
    with open(MODEL_METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Model accuracy: {accuracy:.4f}")

    # Save the model
    joblib.dump(model, MODEL_PATH)
    print(f"Model is saved to {MODEL_PATH}")
    
    return model, metrics

def load_model():
    '''Load the trained model'''
    if not os.path.exists(MODEL_PATH):
        print("Model not found Training an new model!!!")
        model, metrics = train_and_save_model()
        return model
    return joblib.load(MODEL_PATH)

model = load_model()

def validate_input(data, required_features):
    '''validate input data for missing features'''
    missing_features = [feature for feature in required_features if feature not in data]
    if missing_features:
        raise ValueError(f"Missing feature:{','.join(missing_features)}")

# Frontend route
@app.route('/')
def index():
    '''Serve the main dashboard'''
    return render_template('index.html')

@app.route('/api/health')
def health_check():
    '''Health check endpoint'''
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/model-info')
def model_info():
    '''Get model information and metrics'''
    try:
        if os.path.exists(MODEL_METRICS_PATH):
            with open(MODEL_METRICS_PATH, 'r') as f:
                metrics = json.load(f)
        else:
            metrics = {'message': 'No metrics available'}
        
        return jsonify({
            'required_features': REQUIRED_FEATURES,
            'model_type': 'Random Forest Classifier',
            'metrics': metrics
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    '''Real time prediction endpoint for a specific usecase'''
    try:
        data = request.get_json()
        validate_input(data, REQUIRED_FEATURES)
        
        # Convert input into array
        input_data = np.array([data[feature] for feature in REQUIRED_FEATURES]).reshape(1, -1)
        
        # Make prediction and get probability
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)
        
        # Save this prediction into a file
        record = {
            **data, 
            "Prediction": int(prediction[0]),
            "Probability_No_Diabetes": float(probability[0][0]),
            "Probability_Diabetes": float(probability[0][1]),
            "Timestamp": datetime.now().isoformat()
        }
        
        os.makedirs("data", exist_ok=True)
        file_exists = os.path.isfile(REAL_TIME_PREDICTIONS_PATH)
        df = pd.DataFrame([record])
        df.to_csv(REAL_TIME_PREDICTIONS_PATH, mode='a', index=False, header=not file_exists)
        
        return jsonify({
            'prediction': int(prediction[0]),
            'prediction_label': 'Diabetes' if prediction[0] == 1 else 'No Diabetes',
            'probabilities': {
                'no_diabetes': float(probability[0][0]),
                'diabetes': float(probability[0][1])
            },
            'confidence': float(max(probability[0]))
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    '''Batch prediction endpoint'''
    try:
        # Check if file is provided
        if 'file' not in request.files:
            return jsonify({'error':'no files uploaded by user'}), 400
        
        file = request.files['file']
        batch_data = pd.read_csv(file)
        
        # Validating input data
        missing_features = [feature for feature in REQUIRED_FEATURES if feature not in batch_data.columns]
        if missing_features:
            return jsonify({'error':f"Missing features in batch file: {','.join(missing_features)}"}), 400
        
        # Make predictions
        X = batch_data[REQUIRED_FEATURES]
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        # Add predictions and probabilities to batch data
        batch_data['Prediction'] = predictions
        batch_data['Prediction_Label'] = ['Diabetes' if p == 1 else 'No Diabetes' for p in predictions]
        batch_data['Probability_No_Diabetes'] = probabilities[:, 0]
        batch_data['Probability_Diabetes'] = probabilities[:, 1]
        batch_data['Timestamp'] = datetime.now().isoformat()
        
        os.makedirs("data", exist_ok=True)
        batch_data.to_csv(BATCH_PREDICTIONS_PATH, index=False)
        
        return jsonify({
            'message': 'Batch predictions completed successfully',
            'total_predictions': len(batch_data),
            'diabetes_cases': int(sum(predictions)),
            'output_file': BATCH_PREDICTIONS_PATH
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/predictions/history')
def get_prediction_history():
    '''Get prediction history'''
    try:
        history_data = []
        
        # Get real-time predictions
        if os.path.exists(REAL_TIME_PREDICTIONS_PATH):
            rt_df = pd.read_csv(REAL_TIME_PREDICTIONS_PATH)
            rt_df['Type'] = 'Real-time'
            history_data.append(rt_df)
        
        # Get batch predictions
        if os.path.exists(BATCH_PREDICTIONS_PATH):
            batch_df = pd.read_csv(BATCH_PREDICTIONS_PATH)
            batch_df['Type'] = 'Batch'
            history_data.append(batch_df)
        
        if history_data:
            combined_df = pd.concat(history_data, ignore_index=True)
            # Get recent 50 predictions
            recent_predictions = combined_df.tail(50).to_dict('records')
            
            return jsonify({
                'predictions': recent_predictions,
                'total_count': len(combined_df)
            })
        else:
            return jsonify({'predictions': [], 'total_count': 0})
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/statistics')
def get_statistics():
    '''Get prediction statistics'''
    try:
        stats = {}
        
        # Real-time predictions stats
        if os.path.exists(REAL_TIME_PREDICTIONS_PATH):
            rt_df = pd.read_csv(REAL_TIME_PREDICTIONS_PATH)
            stats['realtime'] = {
                'total': len(rt_df),
                'diabetes_cases': int(rt_df['Prediction'].sum()) if 'Prediction' in rt_df.columns else 0,
                'no_diabetes_cases': int(len(rt_df) - rt_df['Prediction'].sum()) if 'Prediction' in rt_df.columns else 0
            }
        else:
            stats['realtime'] = {'total': 0, 'diabetes_cases': 0, 'no_diabetes_cases': 0}
        
        # Batch predictions stats
        if os.path.exists(BATCH_PREDICTIONS_PATH):
            batch_df = pd.read_csv(BATCH_PREDICTIONS_PATH)
            stats['batch'] = {
                'total': len(batch_df),
                'diabetes_cases': int(batch_df['Prediction'].sum()) if 'Prediction' in batch_df.columns else 0,
                'no_diabetes_cases': int(len(batch_df) - batch_df['Prediction'].sum()) if 'Prediction' in batch_df.columns else 0
            }
        else:
            stats['batch'] = {'total': 0, 'diabetes_cases': 0, 'no_diabetes_cases': 0}
        
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/download/<file_type>')
def download_file(file_type):
    '''Download prediction files'''
    try:
        if file_type == 'realtime' and os.path.exists(REAL_TIME_PREDICTIONS_PATH):
            return send_file(REAL_TIME_PREDICTIONS_PATH, as_attachment=True)
        elif file_type == 'batch' and os.path.exists(BATCH_PREDICTIONS_PATH):
            return send_file(BATCH_PREDICTIONS_PATH, as_attachment=True)
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ =='__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)