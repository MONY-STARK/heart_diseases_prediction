from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from base_model.preprocessing import preprocess_data, split_data, load_data
from pathlib import Path
import joblib



class LogisticRegressionModel:

    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)
        self.x_train = self.x_test = self.y_train = self.y_test = None
        self.version = "1.0.0"


    def train(self, data_path):
        data = load_data(data_path)
    #    print("before preprocessed")
    #    data.info()
        data = preprocess_data(data)
    #    print("after preprocessed")
    #    data.info()
    #    return

        self.x_train, self.x_test, self.y_train, self.y_test = split_data(data)
        self.model.fit(self.x_train, self.y_train)


    def evaluate(self):
        y_pred = self.model.predict(self.x_test)
        print("------------------CLassification Report----------------------------")
        print(classification_report(self.y_test, y_pred))
        print("------------------Confusion Matrics--------------------------------")
        print(confusion_matrix(self.y_test, y_pred))




    def save(self, model_path=None):
        if model_path is None:
            model_path = f"saved_models/{self.version}/model.pkl"
            
        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump({
            "model": self.model
        }, path)

