# Digits-Prediction-Neural-Network

This project involves creating a simple neural network in a Jupyter Notebook to predict handwritten digits using the MNIST dataset. The model is built using **Keras** and **TensorFlow** with a **flatten layer** and a **softmax output layer**, applying the **ReLU activation function** for classification.

## Project Overview

- **Goal**: Predict handwritten digits (0-9) using a neural network.
- **Model Architecture**:
  - A flatten layer to convert 28x28 pixel images into a 1D array.
  - Two **dense layers** with 256 neurons each, using the **ReLU activation function**, with a dropout layer in between to prevent overfitting.
  - A single **output layer** with 10 neurons (one for each digit) using the **softmax activation function** for classification.
- **Accuracy**: The model achieved an accuracy of 97.8%.

## How to Run

1. Install the required dependencies:
   ```bash
   pip install tensorflow numpy matplotlib

2. Launch Jupyter Notebook using terminal:
    ```bash
    jupyter notebook
    
Open the provided notebook and follow the steps to train the neural network on the MNIST dataset.


## Steps to Create the Jupyter Notebook

1. **Import necessary libraries**:
   - TensorFlow
   - Keras
   - NumPy
   - Matplotlib

2. **Load the MNIST dataset**:
   - The dataset consists of 60,000 training images and 10,000 test images of handwritten digits.

3. **Preprocess the data**:
   - Flatten the 28x28 pixel images into a 1D array of 784 pixels.
   - Normalize the pixel values (scale them between 0 and 1).

4. **Build the neural network**:
   - Define a sequential model with:
     - A flatten layer to convert the input shape from (28, 28) to a 1D array.
     - Two dense layers with 256 neurons each, using the **ReLU** activation function, with a dropout layer in between to prevent overfitting.
     - One dense output layer with 10 neurons (one for each digit), using the **softmax** activation function.

     **Increased accuracy from 78% to 98%.**

5. **Compile the model**:
   - Use the Adam optimizer.
   - Set the loss function to `sparse_categorical_crossentropy`.
   - Use accuracy as the metric.

6. **Train the model**:
   - Train the model on the training data for 10 epochs.

7. **Evaluate the model**:
   - Evaluate the model performance on the test data.

8. **Make predictions**:
   - Use the trained model to predict digit values from new images.


## Additional Information
**You can do it in Jupyter Notebook as well as through Python code.**
- model.py: This file will train the model and save it in your current directory.
- DigitChecker.py: In this file, you can enter an index to check the predicted digit.
