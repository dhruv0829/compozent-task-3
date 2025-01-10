from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)





# Correct file path
model = joblib.load(r'C:/Users/Dhruv/Documents/housing_price_model.pkl')


# Route for the main page
@app.route('/')
def home():
    return render_template('index.html')

# Route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from the form
        input_data = [float(x) for x in request.form.values()]

        # Convert input data to a NumPy array and reshape it for prediction
        final_data = np.array(input_data).reshape(1, -1)

        # Make prediction using the loaded model
        prediction = model.predict(final_data)

        # Return the result
        return render_template('index.html', prediction_text=f'Predicted Housing Price: ${prediction[0]:,.2f}')
    except Exception as e:
        return jsonify({'error': str(e)})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
