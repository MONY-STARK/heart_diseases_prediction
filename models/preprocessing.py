
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline




def build_preprocessor():

    binary_cols = ["male", "currentSmoker", "BPMeds","prevalentStroke", "prevalentHyp", "diabetes"]

    continuous_cols = ["age", "cigsPerDay", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]

    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    binary_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])

    preprocesser = ColumnTransformer(transformers=[
        ("numeric_pipe", numeric_pipe, continuous_cols),
        ("binary_pipe", binary_pipe, binary_cols)
    ],
    remainder="drop")
    

    return preprocesser

def build_pipeline(model):

    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", model)
        ])
