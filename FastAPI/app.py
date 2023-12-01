import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import numpy as np
import onnxruntime
from fastapi import HTTPException

# Load ONNX model using onnxruntime
model = onnxruntime.InferenceSession('best_model.onnx')
print("ONNX Modèle chargé avec succès.")

app = FastAPI()

class Patient(BaseModel):
    Gender: int
    Age: float
    Major: int
    Year: int
    CGPA: int
    Marriage: int
    Anxiety: int
    Panic: int
    Treatment: int

@app.get("/")
def home():
    return {'message': 'Prediction for Treatment'}

@app.post("/predict_treatment")
async def predict_treatment(patient: Patient):
    input_data = np.array([
        [patient.Gender, patient.Age, patient.Major, patient.Year, patient.CGPA,
         patient.Marriage, patient.Anxiety, patient.Panic, patient.Treatment]
    ], dtype=np.float32)

    # Use the run method for prediction with ONNX model
    pred = model.run(None, {'input': input_data})

    # Assuming a binary classification, use a threshold to get the predicted label
    threshold = 0.5
    prediction = 1 if pred[0][0] > threshold else 0

    if prediction == 1:
        prediction_text = "Traitement recommandé"
    else:
        prediction_text = "Aucun traitement recommandé (Pas de dépression)"

    return {"prediction": prediction_text}

if __name__ == '__main__':
    uvicorn.run("app:app", host='localhost', port=8000, reload=True)
