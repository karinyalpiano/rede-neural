import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

tf.random.set_seed(42)

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 28*28) / 255.0
X_test  = X_test.reshape(-1, 28*28) / 255.0

model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.1)

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Acurácia no teste: {acc:.2%}")

model.save("mnist_mlp.keras")
print("Modelo salvo em mnist_mlp.keras")
