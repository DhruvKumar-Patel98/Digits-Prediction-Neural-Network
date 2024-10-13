from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
(X_train, Y_train) , (X_test, Y_test) = keras.datasets.mnist.load_data()
X_train = X_train/255
X_test = X_test/255
model = keras.models.load_model('handwritten.keras')
y_predicted = model.predict(X_test)
y_predicted[0]
print(np.argmax(y_predicted[0]))
plt.imshow(X_test[0])
plt.show()