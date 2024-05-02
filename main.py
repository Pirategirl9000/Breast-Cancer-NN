import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

input_layer_neurons = 1
hidden_layer_1_neurons = 1

ds_x = pd.read_csv("wdbc.csv")
ds_y = ds_x.pop(0)

(x_train, x_test), (y_train, y_test) = train_test_split(ds_x, ds_y)

biases = np.zeros(shape=(hidden_layer_1_neurons, 1))
weights = np.randn(axis=1, shape=(input_layer_neurons, x_train.length))

def forward_propagation(w, b, x1):
  y1 = np.sum(np.multiply(w[0], x1), b[0])
  x2 = np.sum(1 / (1 + np.exp(y1)) #sigmoid activation
  y2 = np.sum(np.multiply(w[1], x1), b[1])

  return y1, x2, y2

def back_propagation(w, b, x1, y1, x2, y2, rate, y):
  error1 = 0.5 * ((y[0] - y1) * (y[0] - y1)) #MSE
  error2 = 0.5 * ((y[0] - y2) * (y[0] - y2))

  #w1 = w1 + -(error1 * rate * x1)
  w[0] = np.sum(w[0], np.multiply(-1, np.multiply(np.multiply(error1, rate), x1)))
  w[1] = np.sum(w[1], np.multiply(-1, np.multiply(np.multiply(error2, rate), x2)))

  #b1 = b1 + -(error1 * rate)
  b[0] = np.sum(b[0], np.multiply(-1, np.multiply(error1, rate)))
  b[1] = np.sum(b[1], np.multiply(-1, np.multiply(error2, rate)))

  return w, b, error1 + error2

for i in range(100):
  y1, x2, y2 = forward_propagation(weights, biases, x_train)

  weights, biases, loss = back_propagation(weights, biases, x_train, y1, x2, y2, 0.01, y_train)

  if i % 10 == 0:
    print(f"Actual Value: {y_train[i]}, Predicted Value: {y2[i}, loss: {loss}")
  
