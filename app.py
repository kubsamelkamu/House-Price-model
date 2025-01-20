from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('models/best_gradient_boosting.pkl')

# Load the list of features used during training
features = joblib.load('models/features.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the POST request
        data = request.form.to_dict()

        # Handle categorical features dynamically
        categorical_features = ['Alley_Pave']  # Extend this list with other categorical features
        for feature in categorical_features:
            if feature in data:
                data[feature] = 1 if data[feature].strip().lower() == 'yes' else 0

        # Create a DataFrame with all required features
        data_df = pd.DataFrame([data])
        empty_df = pd.DataFrame(columns=features)

        # Fill missing columns with 0 and infer data types
        data_df = pd.concat([empty_df, data_df], ignore_index=True).fillna(0).infer_objects(copy=False)

        # Reorder columns to match the training data
        data_df = data_df[features]

        # Convert appropriate columns to numerical types
        data_df = data_df.astype(float)

        # Make prediction
        prediction = model.predict(data_df)[0]

        # Format the prediction value
        formatted_prediction = "${:,.2f}".format(prediction)

        # Return the result as JSON
        return jsonify({'prediction': formatted_prediction})

    except Exception as e:
        # Return a user-friendly error message
        return jsonify({'error': 'An error occurred during prediction. Please check your input data.'})

if __name__ == "__main__":
    app.run(debug=True)
