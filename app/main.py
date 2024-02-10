# import the required libraries
import uvicorn
from fastapi import FastAPI
import pandas as pd
import pickle
import joblib
import numpy as np
from pydantic import BaseModel
from sklearn.base import BaseEstimator, TransformerMixin
from columntransformer import ConvertColumnsTransformer

# create an instance of FastAPI
app = FastAPI()

# loading the model
model = joblib.load('models/tuned_rf_classifier.pkl')
pipeline = joblib.load('models/preproc_pipeline.pkl')

# Define the column_fct and categorical_cols variables
column_fct = ["ChronicDiseases"]
categorical_cols = ["Employment Type", "GraduateOrNot", "ChronicDiseases", "FrequentFlyer", "EverTravelledAbroad"]
numeric_cols = ["Age", "AnnualIncome", "FamilyMembers"]

class PredictionInput(BaseModel):
    Age: float
    Employment_Type: str
    GraduateOrNot: str
    AnnualIncome: float
    FamilyMembers: float
    ChronicDiseases: float
    FrequentFlyer: str
    EverTravelledAbroad: str

@app.get('/')
def index():
    return {'message': 'Hello world'}

@app.post('/predict')
def predict(data: PredictionInput):
    data = data.dict()
    Age = data['Age']
    Employment_Type = data['Employment_Type']
    GraduateOrNot = data['GraduateOrNot']
    AnnualIncome = data['AnnualIncome']
    FamilyMembers = data['FamilyMembers']
    ChronicDiseases = data['ChronicDiseases']
    FrequentFlyer = data['FrequentFlyer']
    EverTravelledAbroad = data['EverTravelledAbroad']

    # Create a DataFrame from the input values
    data_dict = {
        'Age': [Age],
        'Employment Type': [Employment_Type],
        'GraduateOrNot': [GraduateOrNot],
        'AnnualIncome': [AnnualIncome],
        'FamilyMembers': [FamilyMembers],
        'ChronicDiseases': [ChronicDiseases],
        'FrequentFlyer': [FrequentFlyer],
        'EverTravelledAbroad': [EverTravelledAbroad]
    }

    # Convert dictionary to DataFrame
    query_df = pd.DataFrame(data_dict)

    df = pipeline.transform(query_df)

    # Make the prediction using the loaded model
    prediction = model.predict(df)
    predicted_class = prediction[0]

    if predicted_class == 1:
        prediction_text = "The customer will buy travel insurance."
    else:
        prediction_text = "The customer will not buy travel insurance."

    return prediction_text

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8080)