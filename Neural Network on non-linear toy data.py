
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate a sample dataset
X, y = make_moons(
    n_samples=500,
    noise=0.2,
    random_state=42
)

# Split data into training and testing sets
Xtr, Xte, ytr, yte = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Create the Neural Network model
model = Sequential([
    Dense(8, activation="relu", input_shape=(2,)),
    Dense(1, activation="sigmoid")
])

# Compile the model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Train the model
model.fit(
    Xtr, ytr,
    epochs=30,
    batch_size=16,
    verbose=0
)

# Predict the test data
y_pred = (
    model.predict(Xte, verbose=0) > 0.5
).astype(int).flatten()

# Display test accuracy
print("NN test accuracy:", round(accuracy_score(yte, y_pred), 4))
