from base_model.model import LogisticRegressionModel
from pathlib import Path




if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    data_path = BASE_DIR/"data"/"framingham.csv"
    


    model = LogisticRegressionModel()
    model.train(data_path)
    model.evaluate()
    model.save()
