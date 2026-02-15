from models.model import Model
from keras import Sequential, layers
from keras.layers import Rescaling
from keras.optimizers import Adam, RMSprop

class BasicModel(Model):
    def _define_model(self, input_shape, categories_count):
        self.model = Sequential(
            [
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
                layers.Flatten(),
                layers.Dense(48, activation="relu"),
                layers.Dense(categories_count, activation="softmax"),
            ]
        )

    def _compile_model(self):
        self.model.compile(
            optimizer="rmsprop",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        self.model.learning_rate = 0.001