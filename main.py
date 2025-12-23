from base_model.model import LogisticRegressionModel




if __name__ == "__main__":
    data_path = "/home/stark/Documents/Ai_project/heart_diseases_prediction/framingham.csv"


    model = LogisticRegressionModel()
    model.train(data_path)
    model.evaluate()
    model.save(model_path="./saved_models/base_logistic_model.pkl")
