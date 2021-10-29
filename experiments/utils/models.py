from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf

from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

class Model(ABC):
    def __init__(self, **args) -> None:
        super().__init__()
        self._model = None
        self._epochs = None
        self._batch_size = None
        if args:
            self.set_params(**args)

    @property
    def model(self):
        return self._model

    @property
    def epochs(self):
        return self._epochs

    @property
    def batch_size(self):
        return self._batch_size

    def set_params(self, epochs, batch_size, **parameters):
        self._epochs = epochs
        self._batch_size = batch_size
        self._model = self.build(**parameters)
        pass

    @abstractmethod
    def build(self, **args):
        pass

    def assert_model_built(self):
        assert (
            self._model is not None
        ), "Model has not been built, please call set_params function before training or predicting."

    def fit(self, x_, y):
        self.assert_model_built()
        x = np.expand_dims(x_, axis=-1)
        print(x.shape)
        loss = self._model.fit(x, y, batch_size=self._batch_size, epochs=self._epochs)
        return loss

    def predict(self, x_):
        x = np.expand_dims(x_, axis=-1)
        self.assert_model_built()
        return self._model(x, training=False)


class MLP(Model):
    def build(self, **args):
        return self._mlp(**args)

    def _mlp(self,
        input_shape,
        output_size=1,
        optimizer="adam",
        loss="mae",
        hidden_layers=[32, 16, 8],
        dropout=0.0,
    ):
        inputs = tf.keras.layers.Input(shape=input_shape[-2:])
        x = tf.keras.layers.Flatten()(inputs)  # Convert the 2d input in a 1d array
        for hidden_units in hidden_layers:
            x = tf.keras.layers.Dense(hidden_units)(x)
            x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.Dense(output_size)(x)

        model = tf.keras.Model(inputs=inputs, outputs=x)
        model.compile(optimizer=optimizer, loss=loss)

        return model


class LSTM(Model):
    def build(self, **args):
        return self._lstm(**args)

    def _lstm(self,
        input_shape,
        output_size=1,
        optimizer="adam",
        loss="mae",
        units=64,
        layers=2,
        recurrent_dropout=0,
        return_sequences=False,
        dense_layers=[],
        dense_dropout=0,
    ):
        recurrent_units = [units] * layers

        inputs = tf.keras.layers.Input(shape=input_shape[-2:])
        # LSTM layers
        return_sequences_tmp = return_sequences if len(recurrent_units) == 1 else True
        x = tf.keras.layers.LSTM(
            recurrent_units[0],
            return_sequences=return_sequences_tmp,
            dropout=recurrent_dropout,
        )(inputs)
        for i, u in enumerate(recurrent_units[1:]):
            return_sequences_tmp = (
                return_sequences if i == len(recurrent_units) - 2 else True
            )
            x = tf.keras.layers.LSTM(
                u, return_sequences=return_sequences_tmp, dropout=recurrent_dropout
            )(x)
        # Dense layers
        if return_sequences:
            x = tf.keras.layers.Flatten()(x)
        for hidden_units in dense_layers:
            x = tf.keras.layers.Dense(hidden_units)(x)
            if dense_dropout > 0:
                x = tf.keras.layers.Dropout(dense_dropout)(dense_dropout)
        x = tf.keras.layers.Dense(output_size)(x)

        model = tf.keras.Model(inputs=inputs, outputs=x)
        model.compile(optimizer=optimizer, loss=loss)

        return model


class CNN(Model):
    def build(self, **args):
        return self._cnn(**args)

    def _cnn(self,
        input_shape,
        output_size=1,
        optimizer="adam",
        loss="mae",
        conv_blocks=[[64, 7, 2], [128, 5, 2]],
        dense_layers=[],
        dense_dropout=0.0,
    ):
        conv_layers = [b[0] for b in conv_blocks]
        kernel_sizes = [b[1] for b in conv_blocks]
        pool_sizes = [b[2] for b in conv_blocks]

        assert len(conv_layers) == len(kernel_sizes)
        assert 0 <= dense_dropout <= 1
        inputs = tf.keras.layers.Input(shape=input_shape[-2:])
        # First conv block
        x = tf.keras.layers.Conv1D(
            conv_layers[0], kernel_sizes[0], activation="relu", padding="same"
        )(inputs)
        if pool_sizes[0] and x.shape[-2] // pool_sizes[0] > 1:
            x = tf.keras.layers.MaxPool1D(pool_size=pool_sizes[0])(x)
        # Rest of the conv blocks
        for chanels, kernel_size, pool_size in zip(
            conv_layers[1:], kernel_sizes[1:], pool_sizes[1:]
        ):
            x = tf.keras.layers.Conv1D(
                chanels, kernel_size, activation="relu", padding="same"
            )(x)
            if pool_size and x.shape[-2] // pool_size > 1:
                x = tf.keras.layers.MaxPool1D(pool_size=pool_size)(x)
        # Dense block
        x = tf.keras.layers.Flatten()(x)
        for hidden_units in dense_layers:
            x = tf.keras.layers.Dense(hidden_units)(x)
            if dense_dropout > 0:
                tf.keras.layers.Dropout(dense_dropout)(x)
        x = tf.keras.layers.Dense(output_size)(x)

        model = tf.keras.Model(inputs=inputs, outputs=x)
        model.compile(optimizer=optimizer, loss=loss)
        return model


def RandomForest():
    return RandomForestRegressor()


def XGBoost():
    return MultiOutputRegressor(XGBRegressor())