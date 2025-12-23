from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from base_model.preprocessing import preprocess_data, split_data, load_data
from pathlib import Path
import joblib



class LogisticRegressionModel:

   def __init__(self):
       self.model = LogisticRegression(max_iter=1000)
       self.x_train = self.x_test = self.y_train = self.y_test = None


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
       print(classification_report(self.y_test, y_pred))

   def save(self, model_path="saved_models/logistic_model.pkl"):
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump({
        "model": self.model
    }, path)

