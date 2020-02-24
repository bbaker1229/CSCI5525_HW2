import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers


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


def k_fold_cv(train, test, k, c):
    train_x, train_y, valid_x, valid_y = get_next_train_valid(train, k)
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    clf = SVC(C=c, kernel='linear')
    clf.fit(train_x, train_y.ravel())
    y_pred = clf.predict(train_x)
    y_pred = y_pred.reshape(-1, 1)
    train_acc = (train_y.shape[0] - np.count_nonzero(train_y-y_pred)) / train_y.shape[0]

    y_pred = clf.predict(valid_x)
    y_pred = y_pred.reshape(-1, 1)
    valid_acc = (valid_y.shape[0] - np.count_nonzero(valid_y-y_pred)) / valid_y.shape[0]
    test_x = test[[0, 1]]
    test_y = test[[2]]
    y_pred = clf.predict(test_x)
    y_pred = y_pred.reshape(-1, 1)
    test_acc = (test_y.shape[0] - np.count_nonzero(test_y - y_pred)) / test_y.shape[0]
    return train_acc, valid_acc, test_acc


c = 1000
# Read dataset
data = pd.read_csv('hw2data.csv', header=None)
data = shuffle_data(data)
train_data, test_data = train_test_split(data)
train_data = create_k_train_sets(train_data)

train_acc_lst = []
cv_acc_lst = []
test_acc_lst = []
for c in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
    print(c)
    for i in range(10):
        print(i,)
        train_accuracy, cv_accuracy, test_accuracy = k_fold_cv(train_data, test_data, i, c)
        train_acc_lst.append(train_accuracy)
        cv_acc_lst.append(cv_accuracy)
        test_acc_lst.append(test_accuracy)
    print('\n')
