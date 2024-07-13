import numpy as np
import pandas as pd
import uuid

from sklearn.model_selection import KFold

from .model_training import NeuralNet


class ModelEvaluation:
    def __init__(self) -> None:
        self.df = pd.read_csv("data/training_data.csv")

    def __split_features_and_labels(self) -> None:
        self.X = self.df.iloc[:, 1:].values
        self.y = self.df.iloc[:, 0].values

    def evaluate_cv(self) -> None:
        self.__split_features_and_labels()

        mae_list = []
        rmse_list = []

        kf = KFold(n_splits=5, random_state=42, shuffle=True)
        for train_idx, test_idx in kf.split(self.X):
            X_train, X_test = self.X[train_idx], self.X[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]

            model_id = uuid.uuid4()

            nn_exec = NeuralNet(X_train, X_test, y_train, y_test, model_id)
            mae, mse = nn_exec.execute_training_steps()

            mae_list.append(mae)
            rmse_list.append(np.sqrt(mse))

        print(f"Average MAE = {np.mean(mae_list)}")
        print(f"Average RMSE = {np.mean(rmse_list)}")
