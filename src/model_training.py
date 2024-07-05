import numpy as np
import pickle as pkl

from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from keras import layers, models, utils


# Set a random seed
utils.set_random_seed(42)


class NeuralNet:
    def __init__(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
    ) -> None:
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def __scale_features(self) -> None:
        # Define the scaler
        scaler = StandardScaler()
        self.scaler = scaler.fit(self.X_train)

        # Scale train and test features
        self.scaled_X_train = self.scaler.transform(self.X_train)
        self.scaled_X_test = self.scaler.transform(self.X_test)

    def __build_model(self) -> models.Model:
        model = keras.Sequential(
            [
                layers.Dense(
                    20,
                    activation="relu",
                    input_shape=[self.scaled_X_train.shape[1]],
                    kernel_regularizer=keras.regularizers.L1L2(l1=0.001, l2=0.001),
                ),
                layers.Dense(
                    40,
                    activation="relu",
                    kernel_regularizer=keras.regularizers.L1L2(l1=0.001, l2=0.001),
                ),
                layers.Dense(
                    10,
                    activation="relu",
                    kernel_regularizer=keras.regularizers.L1L2(l1=0.001, l2=0.001),
                ),
                layers.Dense(1),
            ]
        )
        model.compile(loss="mse", optimizer="adam", metrics=["mae", "mse"])

        return model

    def __train_model(self) -> None:
        self.nn_model = self.__build_model()

        # Add an early stopping
        stop_early = keras.callbacks.EarlyStopping(monitor="val_mse", patience=300)
        self.nn_model.fit(
            self.scaled_X_train,
            self.y_train,
            batch_size=self.scaled_X_train.shape[0],
            epochs=3000,
            validation_data=(self.scaled_X_test, self.y_test),
            callbacks=[stop_early],
            verbose=0,
        )

    def __model_metrics(self) -> None:
        _, self.mae, self.mse = self.nn_model.evaluate(self.scaled_X_test, self.y_test)

        print(f"MAE - {self.mae}\nRMSE - {np.sqrt(self.mse)}")

    def __save_model(self) -> None:
        # Fitted model
        models.save_model(
            self.nn_model, "C:/Projetos/fp-neural-net/models/fitted_nn.keras"
        )

        # Fitted scaler
        with open("C:/Projetos/fp-neural-net/models/fitted_scaler.pkl", "wb") as f:
            pkl.dump(self.scaler, f)

        print("Model and scaler were succesfully saved")

    def execute_training_steps(self) -> tuple[float, float]:
        self.__scale_features()
        self.__train_model()
        self.__model_metrics()
        self.__save_model()

        return self.mae, self.mse
