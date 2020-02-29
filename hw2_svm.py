############################################
# Name: Bryan John Baker
# Student ID: bake1358
# Email: bake1358@umn.edu
# Collaborators:
############################################

# Import modules
import numpy as np
import pandas as pd
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
import matplotlib.pyplot as plt


def shuffle_data(df):
    """
    Input a data frame and return the same data frame with it's rows shuffled.
    :param df: Input a pandas data frame.
    :return: A row shuffled data frame.
    """
    row_nums = np.arange(df.shape[0])
    np.random.seed(42)  # Make this repeatable
    np.random.shuffle(row_nums)
    df_shuffled = df.loc[row_nums]
    return df_shuffled


def train_test_split(df, train_frac):
    """
    Split a data frame into a training set and a test set
    by a specified fraction amount for the training set.
    :param df: A pandas data frame
    :param train_frac: The fraction of data to use as
    the training set.
    :return: Two pandas data frames.  A training data frame
    and a test data frame.
    """
    row_nums = np.arange(df.shape[0])
    m = int(df.shape[0] * train_frac)
    df_train = df.loc[row_nums[:m]]
    df_test = df.loc[row_nums[m:]]
    return df_train, df_test


def create_k_train_sets(df, k):
    """
    Input a data frame and the number of folds to split it into.
    Get a dictionary of the data frame split into k groups.
    :param df: A pandas data frame.
    :param k: The number of folds.
    :return: A dictionary of pandas data frames with the keys being the
    fold numbers.
    """
    df = df.reset_index()
    row_nums = np.arange(df.shape[0])
    m = int(df.shape[0] / k)  # Define the batch size for this k value
    # Define a dictionary of row number lists.
    k_rows = {}
    for i in range(k):
        k_rows[i] = list(row_nums[(i * m): ((i + 1) * m)])
    # Define a dictionary of pandas data frames sectioned into k groups
    k_df = {}
    for i in range(k):
        k_df[i] = df.loc[k_rows[i], :]
    return k_df


def get_next_train_valid(data_dict, itr):
    """
    Input a data dictionary with keys being the fold number and the
    fold to get.  Helps for leave group out cross validation.
    :param data_dict: A dictionary of pandas data frames
    with keys being the fold number.
    :param itr: The fold number of leave out and use as the
    validation dataset.
    :return: Four pandas data frames.  One for X values of the training
    set, X values of the validation set, y values of the training
    set, and y values of the validation set.
    """
    first_flag = 1
    for key in data_dict.keys():
        if key != itr:
            if first_flag:
                train = data_dict[key]
                first_flag = 0
            else:
                train = train.append(data_dict[key], ignore_index=True)
        else:
            valid = data_dict[key]
    train_x = train[[0, 1]]
    train_y = train[[2]]
    valid_x = valid[[0, 1]]
    valid_y = valid[[2]]
    return train_x, train_y, valid_x, valid_y


def svmfit(X, y, c):
    """
    Fit an SVM classification model.
    :param X: A pandas data frame of X features.
    :param y: A pandas data frame of y target values.
    :param c: A training parameter.
    :return: A vector of weights for the SVM model.
    """
    X = np.array(X)
    y = np.array(y)
    m = X.shape[0]
    y = y.reshape(-1, 1)
    X_new = y * X
    H = np.dot(X_new, X_new.transpose())

    # cvxopt format
    cvxopt_solvers.options['show_progress'] = False
    P = cvxopt_matrix(H)
    q = cvxopt_matrix(-np.ones((m, 1)))
    G = cvxopt_matrix(np.vstack((np.eye(m) * -1, np.eye(m))))
    h = cvxopt_matrix(np.hstack((np.zeros(m), np.ones(m) * c)))
    A = cvxopt_matrix(y.reshape(1, -1))
    b = cvxopt_matrix(np.zeros(1))

    # Solve
    sol = cvxopt_solvers.qp(P, q, G, h, A, b)
    lambdas = np.array(sol['x'])

    # calculate weights
    weight = np.matmul((y * lambdas).transpose(), X)
    return weight[0]


def predict(X, weight):
    """
    Predict data using an X input data frame and weights
    from a trained SVM model.
    :param X: A pandas data frame of features.
    :param weight: A numpy array of trained weights for an SVM
    classification model.
    :return: A numpy array of y classification predictions.
    """
    X = np.array(X)
    m = X.shape[0]
    X = X.reshape((2, m))
    weight = weight.reshape(-1, 1)
    y_pred = np.sign(np.matmul(weight.transpose(), X))
    return y_pred.transpose()


def k_fold_cv(train, test, k, c):
    """
    Does k fold cross validation on a training set for an SVM
    classification model.
    :param train: A pandas data frame to be used as the training
    dataset.
    :param test: A pandas data frame to be used as the test dataset.
    :param k: The number of folds to use.
    :param c: The modeling parameter to use for SVM.
    :return: The average training accuracy and the validation
    accuracies.  Also the test accuracy
    using these best weights.
    """
    train_folds = create_k_train_sets(train, k)
    weight_lst = []
    train_acc_lst = []
    cv_acc_lst = []
    print("Fold #:")
    for i in range(k):
        print(i, end=" ")
        train_x, train_y, valid_x, valid_y = get_next_train_valid(train_folds, i)
        weight = svmfit(train_x, train_y, c)
        weight_lst.append(weight)
        y_pred = predict(train_x, weight)
        train_acc = (train_y.shape[0] - np.count_nonzero(train_y - y_pred)) / train_y.shape[0]
        train_acc_lst.append(train_acc)
        y_pred = predict(valid_x, weight)
        valid_acc = (valid_y.shape[0] - np.count_nonzero(valid_y - y_pred)) / valid_y.shape[0]
        cv_acc_lst.append(valid_acc)
    best = cv_acc_lst.index(max(cv_acc_lst))
    test_x = test[[0, 1]]
    test_y = test[[2]]
    y_pred = predict(test_x, weight_lst[best])
    test_acc = (test_y.shape[0] - np.count_nonzero(test_y - y_pred)) / test_y.shape[0]
    print("\n")
    return np.mean(train_acc_lst), np.mean(cv_acc_lst), test_acc


# Define the number of folds to use
k = 10
# Read dataset
data = pd.read_csv('hw2data.csv', header=None)
# Shuffle dataset
data = shuffle_data(data)
# Split dataset into training and test datasets
train_data, test_data = train_test_split(data, train_frac=.8)

# Perform k fold cross validation on a list of C parameters
# Collect the average training and cv accuracies and the test
# accuracy using the best parameters
train_acc_lst = []
cv_acc_lst = []
test_acc_lst = []
c_vals = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
for c in c_vals:
    print("C value: " + str(c))
    train_accuracy, cv_accuracy, test_accuracy = k_fold_cv(train_data, test_data, k, c)
    train_acc_lst.append(train_accuracy)
    cv_acc_lst.append(cv_accuracy)
    test_acc_lst.append(test_accuracy)

# Create the training accuracy plot
train_acc_df = pd.DataFrame(list(zip(c_vals, train_acc_lst)), columns=['C', 'Accuracy'])
train_acc_df.plot(x='C', y='Accuracy', logx=True, ylim=(0, 1), title="Plot of training accuracy by C")
plt.show()

# Create the cv accuracy plot
cv_acc_df = pd.DataFrame(list(zip(c_vals, cv_acc_lst)), columns=['C', 'Accuracy'])
cv_acc_df.plot(x='C', y='Accuracy', logx=True, ylim=(0, 1), title="Plot of cv accuracy by C")
plt.show()

# Create the test accuracy plot
test_acc_df = pd.DataFrame(list(zip(c_vals, test_acc_lst)), columns=['C', 'Accuracy'])
test_acc_df.plot(x='C', y='Accuracy', logx=True, ylim=(0, 1), title="Plot of test accuracy by C")
plt.show()
