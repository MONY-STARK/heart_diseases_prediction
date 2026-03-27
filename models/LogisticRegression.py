from sklearn.linear_model import LogisticRegression
from models.base_model import BaseModel
from sklearn.metrics import classification_report, confusion_matrix
from models.preprocessing import build_pipeline
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
import pandas as pd




class LogisticRegressionModel(BaseModel):

    def __init__(self, data:pd.DataFrame, train = True):
        super().__init__()
        self.want_train = train
        self.model = LogisticRegression(max_iter=1000, 
                                        class_weight="balanced")  
        self.data = pd.read_csv(data)

    def __load_data(self):

        X = self.data.drop(columns=["TenYearCHD"])
        y = self.data["TenYearCHD"]
        return X, y
    
    def __split_data(self):
        X, y = self.__load_data()
        return train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

    def train(self):
        if self.want_train:
            self.x_train, self.x_test, self.y_train, self.y_test = self.__split_data()
            
            model = build_pipeline(self.model)
            model.fit(self.x_train, self.y_train)

            print("----- Model Training Done -----")
            self.save_artifact(model)

    def save_artifact(self, model, model_path = None):

        if model_path is None:
            model_path = f"saved_models/{self.model_version}/model.joblib"
        
        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)      
            
        joblib.dump(model, path)


    def evaluate(self, model_path: str, threshold=0.3):
        model = joblib.load(model_path)
        
        self.x_train, self.x_test, self.y_train, self.y_test = self.__split_data()
        
        probs = model.predict_proba(self.x_test)[:, 1]
        preds = (probs >= threshold).astype(int)

        print(classification_report(self.y_test, preds))


   
