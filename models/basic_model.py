from models.model import Model
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.optimizers import RMSprop

class BasicModel(Model):
    def __init__(
        self,
        input_shape,
        categories_count,
        dropout_after_conv_rate=None,
        dropout_after_dense_rate=None,
        learning_rate=0.001,
    ):
        self.dropout_after_conv_rate = dropout_after_conv_rate
        self.dropout_after_dense_rate = dropout_after_dense_rate
        self.learning_rate = learning_rate
        super().__init__(input_shape, categories_count)

    def _define_model(self, input_shape, categories_count):
        model_layers = [
            layers.Input(shape=input_shape),
            Rescaling(1.0 / 255),
            layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(24, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
        ]

        if self.dropout_after_conv_rate is not None:
            model_layers.append(layers.Dropout(self.dropout_after_conv_rate))

        model_layers.extend([
            layers.Flatten(),
            layers.Dense(48, activation="relu"),
        ])

        if self.dropout_after_dense_rate is not None:
            model_layers.append(layers.Dropout(self.dropout_after_dense_rate))

        model_layers.append(layers.Dense(categories_count, activation="softmax"))

        self.model = Sequential(model_layers)

    def _compile_model(self):
        self.model.compile(
            optimizer=RMSprop(learning_rate=self.learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )