#--- Import necessary libraries for training the model
import pandas as pd
import os
import boto3
from dotenv import load_dotenv
#------ import necessary modules for training the model----
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn
import joblib
import pickle


# Load environment variables
load_dotenv()

# Get environment variables
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')

s3 = boto3.client('s3')

s3 = boto3.resource(
    service_name='s3',
    region_name='eu-north-1',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)
# Function to get data from S3 bucket
def get_data_from_s3(bucket_name, file_name):
    obj = s3.Bucket(bucket_name).Object(file_name).get()
    df = pd.read_csv(obj['Body'])
    return df

# Function to save data to CSV
def save_to_csv(df, folder, filename):
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, filename)
    df.to_csv(file_path, index=False)

# Get data from S3 bucket
bucket_name = 'francisbucket1'
file_name = 'TravelInsurancePrediction.csv'
df = get_data_from_s3(bucket_name, file_name)

# Save data to CSV
save_to_csv(df, 'data', 'traveldata.csv')

print(df.info())
# Preprocessing
X = df.drop(["TravelInsurance","Unnamed: 0"], axis=1)
y = df["TravelInsurance"]

print(df['TravelInsurance'].isnull().sum())

# columns
column_fct = ["ChronicDiseases"]

# creating a custom function for converting data types
def to_object_type(df, columns):
    df[columns] = df[columns].astype(str)
    return df

class ConvertColumnsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = to_object_type(X, columns=column_fct)
        return X

# Function to create and fit the preprocessing pipeline
def create_preprocessing_pipeline(column_fct, numeric_cols, categorical_cols):
           
    preprocessor = ColumnTransformer(
        transformers=[
            ('convert_columns', ConvertColumnsTransformer(), column_fct),
            ('scaler', StandardScaler(), numeric_cols),
            ('ohe', OneHotEncoder(), categorical_cols)
        ],
        remainder='passthrough'
    )

    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
    return pipeline

# Function to oversample the minority class using SMOTE
def oversample_minority_class(X_train_preprocessed, y_train):
    smote = SMOTE()
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_preprocessed, y_train)
    return X_train_resampled, y_train_resampled

# Function to fine-tune the best model (Random Forest) using GridSearchCV
def fine_tune_random_forest(X_train_resampled, y_train_resampled):
    grid_search_rf = RandomForestClassifier()

    param_grid = {
        'n_estimators': [25, 50, 100, 150],
        'max_features': ['sqrt', 'log2', None],
        'max_depth': [3, 6, 9],
        'max_leaf_nodes': [3, 6, 9],
    }
    grid_search = GridSearchCV(grid_search_rf, param_grid, cv=3)
    grid_search.fit(X_train_resampled, y_train_resampled)
    params = grid_search.best_params_
    tuned_rf_classifier = RandomForestClassifier(**grid_search.best_params_)
    tuned_rf_classifier.fit(X_train_resampled, y_train_resampled)

    return params, tuned_rf_classifier


def test_model_accuracy(model, X_test_preprocessed, y_test):
    pred_y_rf_tuned = model.predict(X_test_preprocessed)
    accuracy = accuracy_score(y_test, pred_y_rf_tuned)
    print("Accuracy for Tuned Random Forest:", accuracy)

    return accuracy

def log_to_mlflow(tuned_rf_params, tuned_rf_classifier, accuracy):
    with mlflow.start_run():
        mlflow.log_params(tuned_rf_params)
        mlflow.sklearn.log_model(tuned_rf_classifier, "tuned_rf_classifier")
        mlflow.log_metric("accuracy", accuracy)

# Function to save the model and preprocessing pipeline
def save_model_and_pipeline(model, pipeline, model_file_path='models/tuned_rf_classifier.pkl', pipeline_file_path='models/preproc_pipeline.pkl'):
    joblib.dump(model, model_file_path)
    joblib.dump(pipeline, pipeline_file_path)

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the preprocessing pipeline
pipeline = create_preprocessing_pipeline(column_fct=column_fct,
                                         numeric_cols=["Age", "AnnualIncome", "FamilyMembers"],
                                         categorical_cols=["Employment Type", "GraduateOrNot", "ChronicDiseases", "FrequentFlyer", "EverTravelledAbroad"])

X_train_preprocessed = pipeline.fit_transform(X_train)

# Oversample the minority class using SMOTE
X_train_resampled, y_train_resampled = oversample_minority_class(X_train_preprocessed, y_train)

# Fine-tune the best model (Random Forest) using GridSearchCV
tuned_rf_params, tuned_rf_classifier = fine_tune_random_forest(X_train_resampled, y_train_resampled)

# Test the accuracy of the model
X_test_preprocessed = pipeline.transform(X_test)
accuracy = test_model_accuracy(tuned_rf_classifier, X_test_preprocessed, y_test)

# log to mlflow
params = {'max_depth': 3, 'max_features': None, 'max_leaf_nodes': 9, 'n_estimators': 25}

log_to_mlflow(params, tuned_rf_classifier, accuracy)

save_model_and_pipeline(model=tuned_rf_classifier,pipeline=pipeline)
