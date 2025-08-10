from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
import warnings
warnings.filterwarnings("ignore")

# Define the model
inputs = keras.Input(shape=(784,), name="digits")
x = layers.Dense(64, activation="relu", name="dense_1")(inputs)
x = layers.Dense(64, activation="relu", name="dense_2")(x)
outputs = layers.Dense(10, activation="softmax", name="predictions")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

# Plot and save the model architecture
plot_model(model, to_file="model.png", show_shapes=True, show_layer_names=True)
