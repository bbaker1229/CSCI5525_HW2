############################################
# Name: Bryan John Baker
# Student ID: bake1358
# Email: bake1358@umn.edu
# Collaborators:
############################################

# Import packages
import numpy as np
import pandas as pd


def encode_y(dataset):
    """
    Encode multiclass y columns into columns of 1 and 0 for each label.
    :param dataset: A pandas data frame or target labels.
    :return: An encoded data frame.
    """
    first_flag = 1
    for i in np.unique(dataset[[0]]):
        data = (dataset[[0]] == i).astype(int)
        if first_flag:
            y_label_matrix = data
            first_flag = 0
        else:
            y_label_matrix = pd.concat([y_label_matrix, data], axis=1)
    y_label_matrix.columns = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    return y_label_matrix


def sigma(a):
    """
    The sigma function to use for logistic regression.
    :param a: A numpy array.
    :return: A transformed numpy array.
    """
    return (1 / (1 + np.exp(-a)))


def mnist_train(X, y, learning_rate):
    """
    Train the multi class logistic Model using mini batch processing.
    :param X: A pandas data frame with the feature variables.
    :param y: An encoded data frame with the target labels.
    :param learning_rate: The learning rate to use for batch processing.
    :return: The trained weights for the logistic model.
    """
    batch_size = 100
    original_X = X.copy()
    original_y = y.copy()
    d = original_X.shape[1] + 1
    k = len(np.unique(original_y))
    np.random.seed(42)
    w = np.random.rand(k, d)
    for i in range(int(original_X.shape[0] / batch_size)):
        X = original_X[(i * batch_size):((i+1) * batch_size)]
        y = original_y[(i * batch_size):((i+1) * batch_size)]
        y = np.array(encode_y(y))
        add_ones = np.ones((X.shape[0], 1))
        X = np.append(X, add_ones, 1)
        w = w + learning_rate * np.matmul((y.transpose() - sigma(np.matmul(w, X.transpose()))), X)
    return w


def mnist_predict(weight, X):
    """
    Predict labels using the multi class logistic model.
    :param weight: The trained weights for the multi class logistic model.
    :param X: A pandas data frame to use to predict the target labels.
    :return: The a pandas data frame of predicted labels.
    """
    add_ones = np.ones((X.shape[0], 1))
    X = np.append(X, add_ones, 1)
    preds = np.matmul(weight, X.transpose())
    preds = preds.transpose()
    preds = pd.DataFrame(preds).idxmax(axis=1)
    return preds


def con_mat(actual, predicted):
    """
    Create a confusion matrix from the predicted and actual values.
    :param actual: A pandas data frame containing the target values.
    :param predicted: A numpy array containing the predicted values.
    :return: A confusion matrix
    """
    df = actual.copy()
    df_temp = pd.DataFrame(predicted.transpose())
    df['predicted'] = df_temp.iloc[:, 0].values
    df.columns = ['actual', 'predicted']
    cm = pd.crosstab(df['predicted']
                                   , df['actual']
                                   , rownames=['Predicted']
                                   , colnames=['Actual'])
    return cm


def calculate_accuracy(confusion_matrix):
    """
    Calculate the accuracy from the confusion matrix data.
    :param confusion_matrix: A confusion matrix.
    :return: An accuracy value.
    """
    total = 0
    for i in range(confusion_matrix.shape[0]):
        total += confusion_matrix[i][i]
    total = (total * 1.) / (np.sum(np.sum(confusion_matrix)) * 1.)
    return total


# Read data sets
train = pd.read_csv('mnist_train.csv', header=None)
test = pd.read_csv('mnist_test.csv', header=None)

# Prepare X and y training data sets
X = train.drop(0, axis=1)
y = train[[0]]

# Train a multi class logistic model using a mini batch process
learning_rate = 1.0 / X.shape[0]
weight = mnist_train(X, y, learning_rate)
# Save the resulting weights to a file
np.savetxt("hw2_p6_weights.csv", weight, delimiter=",")

# Predict labels on the training data
y_pred = mnist_predict(weight, X)

# Plot the confusion matrix for the training data
print("Training Data Confusion Matrix")
print(con_mat(y, y_pred))
cvm = con_mat(y, y_pred)
# Print the training accuracy
print("Training Accuracy: " + str(calculate_accuracy(cvm)))

# Prepare the X and y test data sets
X = test.drop(0, axis=1)
y = test[[0]]

# Predict the y labels for the test data set
y_pred = mnist_predict(weight, X)

# Plot the confusion matrix for the test data
print("\nTest Data Confusion Matrix")
print(con_mat(y, y_pred))
cvm = con_mat(y, y_pred)
# Print the test accuracy
print("Test Accuracy: " + str(calculate_accuracy(cvm)))
