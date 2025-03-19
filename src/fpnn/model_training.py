from pathlib import Path
from typing import Sequence
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl

from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from tensorflow import keras, random, data
from keras import layers, models, utils

# Set a random seed
utils.set_random_seed(42)
random.set_seed(42)


class NeuralNet:
    def __init__(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        model_id: str,
        epochs: int = 5000,
        l1_penalty: float = 0.001,
        l2_penalty: float = 0.001,
        workers: int = 1,
        *,
        hidden_layers_layout: Sequence[int] = (16, 32, 16),
        activation_functions: str | tuple[str, ...] = "relu"
    ) -> None:
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model_id = model_id
        self.epochs = epochs
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty
        self.workers = workers
        self.hidden_layers_layout = hidden_layers_layout
        if type(activation_functions) is str:
            self.activation_functions = (activation_functions,) * len(self.hidden_layers_layout)
        else:
            msg_layout_compatibility = "The number of activation functions should match the number of hidden layers"
            assert len(activation_functions) == len(self.hidden_layers_layout), msg_layout_compatibility
            assert type(activation_functions) == tuple[str, ...]
            self.activation_functions = activation_functions
            
        # Attributes defined on execution
        self._scaler: None | StandardScaler = None
        self._scaled_X_train: np.ndarray = np.array([])
        self._scaled_X_test: np.ndarray = np.array([])

    def __scale_features(self) -> None:
        # Define the scaler
        scaler = StandardScaler()
        self._scaler = scaler.fit(self.X_train)

        # Scale train and test features
        self._scaled_X_train = self._scaler.transform(self.X_train)
        self._scaled_X_test = self._scaler.transform(self.X_test)

    def __build_model(self) -> models.Model:
        model = keras.Sequential()
        
        # Add input layer to the NN model 
        model.add(layers.Input(shape=(self._scaled_X_train.shape[1],)))
        
        # Add hidden layers to the NN model
        for layer_units, activation_function in zip(self.hidden_layers_layout, self.activation_functions):
            model.add(
                layers.Dense(
                    layer_units, 
                    activation=activation_function,
                    kernel_regularizer=keras.regularizers.L1L2(
                        l1=self.l1_penalty, l2=self.l2_penalty
                    )
                )
            )
        
        # Add output (target prediction) layer
        model.add(layers.Dense(1, activation="linear"))
        
        # Compile the resulting NN model
        model.compile(loss="mse", optimizer="adam", metrics=["mae", "mse"])

        return model

    def __convert_data_from_ndarray_to_dataset(
        self, X, y, *, buffer_size: int = 1024
    ) -> data.Dataset:
        converted_X_y_as_dataset = data.Dataset.from_tensor_slices((X, y))
        converted_X_y_as_dataset = converted_X_y_as_dataset.shuffle(buffer_size=1024)
        converted_X_y_as_dataset = converted_X_y_as_dataset.batch(buffer_size)
        converted_X_y_as_dataset = converted_X_y_as_dataset.prefetch(data.AUTOTUNE)
        return converted_X_y_as_dataset
    
    def __train_model(self) -> None:
        self.nn_model = self.__build_model()

        # Add an early stopping
        stop_early = keras.callbacks.EarlyStopping(monitor="val_mse", patience=500)
        
        # Buffer size compatible with most GPUs (if needed)
        batch_size = 1024 if self._scaled_X_train.shape[0] > 1024 else self._scaled_X_train.shape[0]
        
        # Convert data from numpy arrays to tensorflow.data.Dataset
        train_dataset = self.__convert_data_from_ndarray_to_dataset(
            self._scaled_X_train, self.y_train, buffer_size=batch_size
        )
        validation_dataset = self.__convert_data_from_ndarray_to_dataset(
            self._scaled_X_test, self.y_test, buffer_size=batch_size
        )
        
        # Model training
        self.nn_model.fit(
            train_dataset,
            epochs=self.epochs,
            validation_data=validation_dataset,
            callbacks=[stop_early],
            verbose=0,
            workers=self.workers,
            use_multiprocessing=True if self.workers != 1 else False
        )

    def __model_metrics(self) -> None:
        _, self.mae, self.mse = self.nn_model.evaluate(self._scaled_X_test, self.y_test)

        print(f"MAE - {self.mae}\nRMSE - {np.sqrt(self.mse)}")

    def __plot_predicted_vs_experimental(self) -> None:
        # Get predicted values
        y_pred = self.nn_model.predict(self._scaled_X_test)

        # Calculate R²
        r2 = r2_score(self.y_test, y_pred)

        # Plot the scatter plot
        plt.scatter(self.y_test, y_pred, label=f"R² = {r2:.4f}")
        plt.xlabel("Experimental values - (K)")
        plt.ylabel("Predicted values - (K)")
        plt.axis("equal")
        plt.axis("square")
        plt.xlim([250, plt.xlim()[1]])
        plt.ylim([250, plt.ylim()[1]])

        # Plot ideal line - Predicted value equal to Experimental
        plt.plot(
            [250, max(plt.xlim()[1], plt.ylim()[1])],
            [250, max(plt.xlim()[1], plt.ylim()[1])],
            linestyle="--",
            color="gray",
            label="Ideal line",
        )

        plt.legend(loc="upper left")
        plt.show()

    def __save_model(self, save_model_path: Path) -> None:
        # Fitted model
        models.save_model(
            self.nn_model, save_model_path / f"{self.model_id}.keras"
        )

        # Fitted scaler
        with open(
            save_model_path / f"scaler_{self.model_id}.pkl", "wb"
        ) as f:
            pkl.dump(self._scaler, f)

        print("Model and scaler were succesfully saved")

    def execute_training_steps(self, *, save_model_path: Path | None = None) -> tuple[float, float]:
        self.__scale_features()
        self.__train_model()
        self.__model_metrics()
        self.__plot_predicted_vs_experimental()
        if save_model_path is not None:
            self.__save_model(save_model_path)

        return self.mae, self.mse
