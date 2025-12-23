import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split



def load_data(data_path):
    path = Path(data_path)
    if path.suffix.lower() != ".csv":
        raise ValueError("File format is not CSV")


    return pd.read_csv(path)


def preprocess_data(dataframe: pd.DataFrame):
    
    dataframe = dataframe.drop(columns=["education"])

    dataframe =  dataframe.fillna({
        "cigsPerDay": dataframe["cigsPerDay"].median(),
        "BPMeds": dataframe["BPMeds"].mode()[0],
        "totChol": dataframe["totChol"].median(),
        "BMI": dataframe["BMI"].median(),
        "heartRate": dataframe["heartRate"].median(),
        "glucose": dataframe["glucose"].median(),
    })
    return dataframe


def split_data(dataframe):
    X = dataframe.drop(columns=["TenYearCHD"])
    y = dataframe["TenYearCHD"]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    x_test.to_csv("X_test.csv", index = False)
    y_test.to_csv("y_test.csv", index = False)

    return x_train, x_test, y_train,y_test


