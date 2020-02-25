############################################
# Name: Bryan John Baker
# Student ID: bake1358
# Email: bake1358@umn.edu
# Collaborators:
############################################

# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def encode_y(dataset):
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


# Sigma function
def sigma(a):
    """
    Define the sigma function to use for logistic regression.
    """
    return (1 / (1 + np.exp(-a)))


def mnist_train(X, y, learning_rate):
    """
    Train the Model
    :param X:
    :param y:
    :param learning_rate:
    :return:
    """
    batch_size = 1000
    original_X = X.copy()
    original_y = y.copy()
    d = original_X.shape[1] + 1
    k = len(np.unique(original_y))
    np.random.seed(42)
    w = np.random.rand(k, d)
    for i in range(int(original_X.shape[0] / batch_size)):
        #print((i * batch_size), ((i+1) * batch_size))
        X = original_X[(i * batch_size):((i+1) * batch_size)]
        y = original_y[(i * batch_size):((i+1) * batch_size)]
        y = np.array(encode_y(y))
        add_ones = np.ones((X.shape[0], 1))
        X = np.append(X, add_ones, 1)
        # eta = 1.0 / X.shape[0]
        w = w + learning_rate * np.matmul((y.transpose() - sigma(np.matmul(w, X.transpose()))), X)
    return w


def mnist_predict(weight, X):
    """
    Predict labels using the model.
    :param weight:
    :param X:
    :return:
    """
    add_ones = np.ones((X.shape[0], 1))
    X = np.append(X, add_ones, 1)
    #preds = sigma(np.matmul(weight, X.transpose()))
    preds = np.matmul(weight, X.transpose())
    preds = preds.transpose()
    preds = pd.DataFrame(preds).idxmax(axis=1)
    return preds


# Create Confusion Matrix
def con_mat(actual, predicted):
    """
    Create a confusion matrix for the logistic regression data.
    """
    df = actual.copy()
    df_temp = pd.DataFrame(predicted.transpose())
    df['predicted'] = df_temp.iloc[:,0].values
    df.columns = ['actual', 'predicted']
    cm = pd.crosstab(df['predicted']
                                   , df['actual']
                                   , rownames=['Predicted']
                                   , colnames=['Actual'])
    return cm


def calculate_accuracy(confusion_matrix):
    total = 0
    for i in range(confusion_matrix.shape[0]):
        total += confusion_matrix[i][i]
    total = (total * 1.) / (np.sum(np.sum(cvm)) * 1.)
    return total


train = pd.read_csv('mnist_train.csv', header=None)
test = pd.read_csv('mnist_test.csv', header=None)

X = train.drop(0, axis=1)
y = train[[0]]

learning_rate = 1.0 / X.shape[0]
weight = mnist_train(X, y, learning_rate)
np.savetxt("hw2_p6_weights.csv", weight, delimiter=",")
y_pred = mnist_predict(weight, X)

print("Training Data Confusion Matrix")
print(con_mat(y, y_pred))
cvm = con_mat(y, y_pred)
print("Training Accuracy: " + str(calculate_accuracy(cvm)))

X = test.drop(0, axis=1)
y = test[[0]]
y_pred = mnist_predict(weight, X)
print("\nTest Data Confusion Matrix")
print(con_mat(y, y_pred))
cvm = con_mat(y, y_pred)
print("Test Accuracy: " + str(calculate_accuracy(cvm)))

# multi-class logistic regression
# Use mini-batches
# Save final trained weights in a file
