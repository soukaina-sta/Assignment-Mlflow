from flask import Flask, render_template, request
import onnxruntime  # Add this line
import numpy as np

# Load the ONNX model
onnx_model = onnxruntime.InferenceSession('best_model.onnx')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = request.form.to_dict()

    # Convert the values to float
    features = [float(features[var]) for var in features]

    final_features = np.array(features).reshape(1, -1)

    # Make predictions using the ONNX model
    input_name = onnx_model.get_inputs()[0].name
    prediction = onnx_model.run(None, {input_name: final_features.astype(np.float32)})

    # Assuming your model output is a single value
    prediction_text = "Great news! You're not diagnosed with depression." if prediction[0] == 0 else "Please consult a healthcare professional. There's a possibility of depression."

    return render_template('index.html', prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
