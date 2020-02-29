############################################
# Name: Bryan John Baker
# Student ID: bake1358
# Email: bake1358@umn.edu
# Collaborators:
############################################

# Import packages
import numpy as np
import pandas as pd


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
    train_x = train.drop('Unnamed: 0', axis=1)
    train_x = train_x.drop('y', axis=1)
    train_x = train_x.drop('index', axis=1)
    train_y = train['y']
    encoded_train_y = encode_y(train_y)
    valid_x = valid.drop('Unnamed: 0', axis=1)
    valid_x = valid_x.drop('y', axis=1)
    valid_x = valid_x.drop('index', axis=1)
    valid_y = valid['y']
    encoded_valid_y = encode_y(valid_y)
    return train_x, encoded_train_y, valid_x, encoded_valid_y


def encode_y(dataset):
    """
    Encode multiclass y columns into columns of 1 and 0 for each label.
    :param dataset: A pandas data frame or target labels.
    :return: An encoded data frame.
    """
    first_flag = 1
    for i in np.unique(dataset):
        data = (dataset == i).astype(int)
        if first_flag:
            y_label_matrix = data
            first_flag = 0
        else:
            y_label_matrix = pd.concat([y_label_matrix, data], axis=1)
    y_label_matrix.columns = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    return y_label_matrix


def rbf_kernel(x, y, sig):
    """
    Define the radial basis function kernel.
    :param x: Parameter 1, for prediction this is the X
    features.
    :param y: Parameter 2, for prediction will be the training
    X features.
    :param sig: sigma hyper parameter
    :return: Kernel values
    """
    return np.exp(-(np.linalg.norm(x-y)**2) / (2 * sig**2))


def mnist_svm_train(X, y, c, sig):
    """
    Train an SVM classifier using a radial basis function.
    :param X: A pandas data frame of the training features.
    :param y: A pandas data frame of the training targets.
    :param c: The C SVM hyperparameter.
    :param sig: The rbf hyperparameter.
    :return: A numpy array of alphas to be used for predictions.
    """
    X = np.array(X)
    m = X.shape[0]
    lambda_array = []
    for z in range(y.shape[1]):
        y_k = y[str(z+1)]
        y_k = np.array(y_k)
        y_k = y_k.reshape(-1, 1)
        K = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                K[i, j] = rbf_kernel(X[i], X[j], sig)
        H = y_k.transpose() * y_k * K

        # cvxopt format
        cvxopt_solvers.options['show_progress'] = False
        P = cvxopt_matrix(H)
        q = cvxopt_matrix(-np.ones((m, 1)))
        G = cvxopt_matrix(np.vstack((np.eye(m) * -1, np.eye(m))))
        h = cvxopt_matrix(np.hstack((np.zeros(m), np.ones(m) * c)))
        A = cvxopt_matrix(y_k.reshape(1, -1) * 1.)
        b = cvxopt_matrix(np.zeros(1))

        # Solve
        sol = cvxopt_solvers.qp(P, q, G, h, A, b)
        lambdas = np.array(sol['x'])
        lambda_array.append(lambdas)

    return lambda_array


def mnist_svm_predict(test_X, train_X, train_y, alpha, sig):
    """
    Predict using an rbf SVM classifier.
    :param test_X: A pandas data frame for test features.
    :param train_X: A pandas data frame for the features
    used in the training dataset.
    :param train_y: A pandas data frame for the targets
    used in the training dataset.
    :param alpha: A numpy array of alpha values from the
    trained rbf SVM classifier.
    :param sig: The rbf hyperparameter.
    :return: A numpy array of y classification predictions.
    """
    X = np.array(train_X)
    X_new = np.array(test_X)
    X_new_norm = np.sum(X_new ** 2, axis=-1)
    X_norm = np.sum(X ** 2, axis=-1)
    for z in range(train_y.shape[1]):
        y_k = train_y[str(z+1)]
        y_k = np.array(y_k)
        y_k = y_k.reshape(-1, 1)

        preds_k = np.matmul((alpha[z] * (np.exp(-(X_new_norm[:,None] + X_norm[None,:] - 2 * np.dot(X_new, X.T)) / (2.0 * sig**2))).T).T, y_k)
        preds_k = pd.DataFrame(np.sign(preds_k))
        if z == 0:
            preds = preds_k
        else:
            preds[z] = preds_k
    preds.columns = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    preds = preds.astype(int)
    preds = np.abs(preds)
    return preds.idxmax(axis=1)


def k_fold_cv(train, test, k, c, sig):
    """
    Does k fold cross validation on a training set for an rbf SVM
    classification model.
    :param train: A pandas data frame to be used as the training
    dataset.
    :param test: A pandas data frame to be used as the test dataset.
    :param k: The number of folds to use.
    :param c: The modeling parameter to use for SVM.
    :param sig: The rbf hyperparameter
    :return: The average training accuracy and the validation
    accuracies.  Also the test accuracy
    using these best weights.
    """
    train_folds = create_k_train_sets(train, k)
    alpha_lst = []
    train_acc_lst = []
    cv_acc_lst = []
    print("Fold #:")
    for i in range(k):
        print(i, end=" ")
        train_x, train_y, valid_x, valid_y = get_next_train_valid(train_folds, i)
        alpha = mnist_svm_train(train_x, train_y, c, sig)
        alpha_lst.append(alpha)
        y_pred = mnist_svm_predict(train_x, train_x, train_y, alpha, sig)
        train_y_values = train_y.idxmax(axis=1).astype(int)
        train_y_values = train_y_values.reset_index()
        train_y_values = train_y_values.drop('index', axis=1)
        y_pred = y_pred.astype(int)
        y_pred = pd.DataFrame(y_pred)
        train_acc = (train_y.shape[0] - np.count_nonzero(train_y_values - y_pred)) / train_y.shape[0]
        train_acc_lst.append(train_acc)
        y_pred = mnist_svm_predict(valid_x, train_x, train_y, alpha, sig)
        valid_y_values = valid_y.idxmax(axis=1).astype(int)
        valid_y_values = valid_y_values.reset_index()
        valid_y_values = valid_y_values.drop('index', axis=1)
        y_pred = y_pred.astype(int)
        y_pred = pd.DataFrame(y_pred)
        valid_acc = (valid_y.shape[0] - np.count_nonzero(valid_y_values - y_pred)) / valid_y.shape[0]
        cv_acc_lst.append(valid_acc)
    best = cv_acc_lst.index(max(cv_acc_lst))
    test_x = test.drop('Unnamed: 0', axis=1)
    test_x = test_x.drop('y', axis=1)
    test_y = test['y']
    test_y = pd.DataFrame(np.array(test_y))
    y_pred = mnist_svm_predict(test_x, train_x, train_y, alpha_lst[best], sig)
    y_pred = y_pred.astype(int)
    y_pred = pd.DataFrame(y_pred)
    test_acc = (test_y.shape[0] - np.count_nonzero(test_y - y_pred)) / test_y.shape[0]
    print("\n")
    return np.mean(train_acc_lst), np.mean(cv_acc_lst), test_acc


def con_mat(actual, predicted):
    """
    Create a confusion matrix from the predicted and actual values.
    :param actual: A pandas data frame containing the target values.
    :param predicted: A numpy array containing the predicted values.
    :return: A confusion matrix
    """
    df = actual.copy()
    df = pd.DataFrame(df)
    df_temp = pd.DataFrame(predicted)
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
        total += confusion_matrix[i+1][i+1]
    total = (total * 1.) / (np.sum(np.sum(confusion_matrix)) * 1.)
    return total


# Functions from Problem 6 are listed below:
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
    batch_size = 10
    original_X = X.copy()
    original_y = y.copy()
    d = original_X.shape[1] + 1
    k = original_y.shape[1]
    np.random.seed(42)
    w = np.random.rand(k, d)
    for i in range(int(original_X.shape[0] / batch_size)):
        X = original_X[(i * batch_size):((i+1) * batch_size)]
        y = original_y[(i * batch_size):((i+1) * batch_size)]
        y = np.array(y)
        add_ones = np.ones((X.shape[0], 1))
        X = np.append(X, add_ones, 1)
        w = w + learning_rate * np.matmul((y.transpose() - sigma(np.matmul(w, X.transpose())).transpose()), X)
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
    preds = pd.DataFrame(preds)
    preds.columns = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    preds = preds.idxmax(axis=1)
    preds = preds.astype(int)
    return preds


# Read data sets
train = pd.read_csv('mfeat_train.csv')
test = pd.read_csv('mfeat_test.csv')

# Prepare X and y training data sets
train = shuffle_data(train)
train_x = train.drop('Unnamed: 0', axis=1)
train_x = train_x.drop('y', axis=1)
train_y = train['y']
encoded_train_y = encode_y(train_y)

# Prepare X and y test data sets
test_x = test.drop('Unnamed: 0', axis=1)
test_x = test_x.drop('y', axis=1)
test_y = test['y']
encoded_test_y = encode_y(test_y)

# Multi class SVM
# Define the number of folds to use
k = 10
# Perform k fold cross validation on a list of C parameters
# Collect the average training and cv accuracies and the test
# accuracy using the best parameters
train_acc_dict = {}
cv_acc_dict = {}
test_acc_dict = {}
c_vals = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
# c_vals = [0.01, 0.1, 1, 10]
sig_vals = c_vals
for c in c_vals:
    print("C value: " + str(c))
    train_acc_lst = []
    cv_acc_lst = []
    test_acc_lst = []
    for sig in sig_vals:
        print("Sigma value: " + str(sig))
        train_accuracy, cv_accuracy, test_accuracy = k_fold_cv(train, test, k, c, sig)
        train_acc_lst.append(train_accuracy)
        cv_acc_lst.append(cv_accuracy)
        test_acc_lst.append(test_accuracy)
    train_acc_dict[c] = train_acc_lst
    cv_acc_dict[c] = cv_acc_lst
    test_acc_dict[c] = test_acc_lst

# Multi class logistic
# Train a multi class logistic model using a mini batch process
learning_rate = 1.0 / train_x.shape[0]
weight = mnist_train(train_x, encoded_train_y, learning_rate)
# Predict labels on the training data
y_pred = mnist_predict(weight, train_x)
# Plot the confusion matrix for the training data
print("Training Data Confusion Matrix")
print(con_mat(train_y, y_pred))
cvm = con_mat(train_y, y_pred)
# Print the training accuracy
print("Training Accuracy: " + str(calculate_accuracy(cvm)))
# Predict the y labels for the test data set
y_pred = mnist_predict(weight, test_x)
# Plot the confusion matrix for the test data
print("\nTest Data Confusion Matrix")
print(con_mat(test_y, y_pred))
cvm = con_mat(test_y, y_pred)
# Print the test accuracy
print("Test Accuracy: " + str(calculate_accuracy(cvm)))
