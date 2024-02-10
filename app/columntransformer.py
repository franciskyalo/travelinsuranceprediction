from sklearn.base import BaseEstimator, TransformerMixin

column_fct =["ChronicDiseases"]
# creating a custom function for converting data types 

def to_object_type(df, columns):
    df[columns] = df[columns].astype(str)
    return df

# create a custom class for converting the chronic diseases to a factor variable


class ConvertColumnsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = to_object_type(X, columns=column_fct)
        return X