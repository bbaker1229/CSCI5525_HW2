############################################
# Name: Bryan John Baker
# Student ID: bake1358
# Email: bake1358@umn.edu
# Collaborators:
############################################

# Import modules
import numpy as np
import pandas as pd
from cvxopt.blas import dot
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
import matplotlib.pyplot as plt


def shuffle_data(df):
    """
    """
    row_nums = np.arange(df.shape[0])
    np.random.seed(42)
    np.random.shuffle(row_nums)
    df_shuffled = df.loc[row_nums]
    return df_shuffled


def train_test_split(df):
    """
    """
    row_nums = np.arange(df.shape[0])
    df_test = df.loc[row_nums[:400]]
    df_train = df.loc[row_nums[400:]]
    return df_train, df_test


def create_k_train_sets(df):
    """
    """
    df = df.reset_index()
    row_nums = np.arange(df.shape[0])
    k_rows = {0: list(row_nums[:160])
              , 1: list(row_nums[160:320])
              , 2: list(row_nums[320:480])
              , 3: list(row_nums[480:640])
              , 4: list(row_nums[640:800])
              , 5: list(row_nums[800:960])
              , 6: list(row_nums[960:1120])
              , 7: list(row_nums[1120:1280])
              , 8: list(row_nums[1280:1440])
              , 9: list(row_nums[1440:])}
    k_df = {0: df.loc[k_rows[0], :]
            , 1: df.loc[k_rows[1], :]
            , 2: df.loc[k_rows[2], :]
            , 3: df.loc[k_rows[3], :]
            , 4: df.loc[k_rows[4], :]
            , 5: df.loc[k_rows[5], :]
            , 6: df.loc[k_rows[6], :]
            , 7: df.loc[k_rows[7], :]
            , 8: df.loc[k_rows[8], :]
            , 9: df.loc[k_rows[9], :]}
    return k_df


# Function to create training and validation sets
def get_next_train_valid(data_dict, itr):
    """
    Create training and validation sets by inputing a fold index.
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
    weight = ((y * lambdas).transpose() @ X).reshape(-1, 1)
    S = (lambdas > 1e-4).flatten()
    b = y[S] - np.dot(X[S], weight)
    weight = np.append(weight, b[0])
    return weight


def predict(X, weight):
    X = np.array(X)
    m = X.shape[0]
    X = np.append(X, np.ones(m))
    X = X.reshape((3, m))
    weight = weight.reshape(-1, 1)
    y_pred = np.sign(np.matmul(weight.transpose(), X))
    return y_pred.transpose()


def k_fold_cv(train, test, k, c):
    # Create training and validation sets from train
    # Use Leave group out CV
    # Find weight for training data
    # Find accuracy on training data
    # Find accuracy on validation data
    # Choose best accuracy weights
    # Use these weights for the test data set
    # Report the average training accuracy
    # Report the average validation accuracy
    # Report the test accuracy
    train_x, train_y, valid_x, valid_y = get_next_train_valid(train, k)
    weight = svmfit(train_x, train_y, c)
    y_pred = predict(train_x, weight)
    train_acc = (train_y.shape[0] - np.count_nonzero(train_y-y_pred)) / train_y.shape[0]

    y_pred = predict(valid_x, weight)
    valid_acc = (valid_y.shape[0] - np.count_nonzero(valid_y-y_pred)) / valid_y.shape[0]
    test_x = test[[0, 1]]
    test_y = test[[2]]
    y_pred = predict(test_x, weight)
    test_acc = (test_y.shape[0] - np.count_nonzero(test_y - y_pred)) / test_y.shape[0]
    return train_acc, valid_acc, test_acc


c = 0.01
# Read dataset
data = pd.read_csv('hw2data.csv', header=None)
data = shuffle_data(data)
train_data, test_data = train_test_split(data)
train_data = create_k_train_sets(train_data)

train_acc_dict = {}
cv_acc_dict = {}
test_acc_dict = {}
for c in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
    print(c)
    train_acc_lst = []
    cv_acc_lst = []
    test_acc_lst = []
    for i in range(10):
        print(i, end=", ")
        train_accuracy, cv_accuracy, test_accuracy = k_fold_cv(train_data, test_data, i, c)
        train_acc_lst.append(train_accuracy)
        cv_acc_lst.append(cv_accuracy)
        test_acc_lst.append(test_accuracy)
    train_acc_dict[c] = train_acc_lst
    cv_acc_dict[c] = cv_acc_lst
    test_acc_dict[c] = test_acc_lst
    print('\n')

delta_list = []
for key in train_acc_dict.keys():
    delta_list.append(np.mean(train_acc_dict[key]))
keys = train_acc_dict.keys()
df = pd.DataFrame(list(zip(keys, delta_list)), columns =['C', 'Accuracy'])

# Create the plot
df.plot(x='C', y='Accuracy', logx=True, ylim=(0, 1), title="Plot of training accuracy by C")
plt.show()

delta_list = []
for key in cv_acc_dict.keys():
    delta_list.append(np.mean(cv_acc_dict[key]))
keys = cv_acc_dict.keys()
df = pd.DataFrame(list(zip(keys, delta_list)), columns =['C', 'Accuracy'])

# Create the plot
df.plot(x='C', y='Accuracy', logx=True, ylim=(0, 1), title="Plot of cv accuracy by C")
plt.show()

delta_list = []
for key in test_acc_dict.keys():
    delta_list.append(np.mean(test_acc_dict[key]))
keys = test_acc_dict.keys()
df = pd.DataFrame(list(zip(keys, delta_list)), columns =['C', 'Accuracy'])

# Create the plot
df.plot(x='C', y='Accuracy', logx=True, ylim=(0, 1), title="Plot of test accuracy by C")
plt.show()
