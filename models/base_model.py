
class BaseModel:

    def __init__(self, model_verion = "2.1.0"):
        self.model_version = model_verion
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None