

class Pipeline:
    """Base class for pipelines"""
    def __init__(self):
        self.pipe = []

    def fit(self, X, Y=None):
        """
        Method to prepare or preprocess data and/or pipeline parameters,
        takes the training data as arguments, which can be one array X,
        or two arrays X and Y (for example, in the case of supervised learning).
        """
        for i in range(len(self.pipe)):
            self.pipe[i].fit(X, Y)
        return self

    def transform(self, X):
        """
        Implements filtering or modifying the data X
        """
        for i in range(len(self.pipe)):
            step = self.pipe[i]
            X = step[1].transform(X)
        return X

    def init_training(self):
        pass


class PipelineStage:

    def fit(self, X, Y=None):
        return self

    def transform(self, X):
        return X
