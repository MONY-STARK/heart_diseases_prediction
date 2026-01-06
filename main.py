
from pathlib import Path
from models.RandomForest import RandomForestClassification
from models.LogisticRegression import LogisticRegressionModel




if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    data_path = BASE_DIR/"data"/"framingham.csv"
    


    model = LogisticRegressionModel(data_path)
    # model.train()
    
    model.evaluate("saved_models/2.1.0/model.joblib", threshold=0.25)
