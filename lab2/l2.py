import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Read data from file
data = np.genfromtxt('data_banknote_authentication.txt', delimiter=',')
X, y = data[:, :-1], data[:, -1]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define loss functions
def logistic_loss(y_true, y_pred):
    return tf.nn.softplus(-y_true * y_pred)

def adaboost_loss(y_true, y_pred):
    return tf.exp(-y_true * y_pred)

def binary_crossentropy(y_true, y_pred):
    return -(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))

# Function for training and evaluating models
def train_and_evaluate(loss_function, X_train, y_train, X_test, y_test, epochs=100):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(1, activation='sigmoid', use_bias=True)
    ])

    model.compile(optimizer='adam', loss=loss_function, metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_test, y_test))

    return model, history.history['loss'], history.history['val_loss']

loss_functions = [logistic_loss, adaboost_loss, binary_crossentropy]

models = []
train_losses = []
test_losses = []

for loss_function in loss_functions:
    model, train_loss, test_loss = train_and_evaluate(loss_function, X_train, y_train, X_test, y_test)
    models.append(model)
    train_losses.append(train_loss)
    test_losses.append(test_loss)

# Create learning curves plots
plt.figure(figsize=(12, 6))
for i, loss_function in enumerate(loss_functions):
    plt.plot(train_losses[i], label=f"Train Loss ({loss_function.__name__})")
    plt.plot(test_losses[i], label=f"Test Loss ({loss_function.__name__})")

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Curves")
plt.legend()
plt.grid(True)
plt.show()

# Compare classification quality using accuracy metric
accuracies = [accuracy_score(y_test, (model.predict(X_test) > 0.5).astype(int)) for model in models]

for i, loss_function in enumerate(loss_functions):
    print(f"Accuracy ({loss_function.__name__}): {accuracies[i]}")