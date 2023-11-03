# Intro Machine-Learning-Projects

## Project 1 Handwritten Digit Recognition with TensorFlow

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

## Project 2 Titanic Survival Prediction with TensorFlow

This code demonstrates a basic example of predicting the survival of passengers on the Titanic using TensorFlow. The dataset contains various features about passengers, and the goal is to build a machine learning model to predict survival.

### Key Features

- **Data Loading**: The code loads Titanic dataset for training and evaluation from Google Cloud Storage using Pandas.

- **Feature Columns**: It prepares feature columns for the model, including both categorical and numeric columns. The categorical columns are processed to handle vocabulary lists.

- **Input Function**: An input function is created for both training and evaluation datasets, which are used by TensorFlow to manage the data during training.

- **Linear Estimator**: The model is created using TensorFlow's Linear Classifier estimator. It's a simple linear model suitable for binary classification tasks like predicting survival.

- **Training and Evaluation**: The code trains the model on the training dataset and evaluates it on the test dataset. The result is printed, which typically includes accuracy.

- **Prediction**: After training, the model is used to make predictions on the test dataset, and the predicted probabilities are plotted as a histogram.

### Dependencies

- TensorFlow
- Numpy
- Pandas
- Matplotlib

### How to Run

You can run this code in a Jupyter Notebook or any Python environment that supports TensorFlow. Ensure that you have the required dependencies installed.

Feel free to modify and extend the code for your own projects and experiments.





