# Intro Machine-Learning-Projects

## Handwritten Digit Recognition with TensorFlow

This code demonstrates a simple handwritten digit recognition model using TensorFlow. It utilizes the MNIST dataset, which contains images of handwritten digits (0-9) and their corresponding labels.

### Key Features

- **Dataset**: The MNIST dataset is loaded and divided into training and test sets. Images are normalized to values between 0 and 1.

- **Model**: A neural network model is constructed with three layers: an input layer, a hidden layer with ReLU activation, and an output layer with a softmax activation.

- **Training**: The model is trained using the Adam optimizer and sparse categorical cross-entropy loss. It's trained for 45 epochs.

- **Evaluation**: The model's accuracy is evaluated on the test set.

- **Prediction**: The model is used to make predictions on test images, and the results are displayed, including a visual representation of the test image and the predicted digit.

### Dependencies

- TensorFlow
- Numpy
- Matplotlib

### How to Run

You can run this code in a Jupyter Notebook or any Python environment that supports TensorFlow. Ensure that you have the required dependencies installed.

Feel free to modify and extend the code for your own projects and experiments.


