"""
利用矩阵进行训练
"""
import numpy as np

def matrix_train(train_label, train_cut, test_cut, vectorizer):
    """
    利用矩阵进行训练
    :param train_label: list[str]
    :param train_cut: list[str]
    :param test_cut: list[str]
    :param vectorizer: Vectorizer(矩阵)
    :return: X_train, Y_train, X_test
    """
    vectorizer.fit(train_cut + test_cut)

    X_train = vectorizer.transform(train_cut).toarray()
    Y_train = np.array(train_label.tolist())
    print('{}X_train shape -->{}{}\n'.format('-'*10, X_train.shape, '-'*10))

    X_test = vectorizer.transform(test_cut).toarray()
    print('{}X_test shape -->{}{}\n'.format('-'*10, X_test.shape, '-'*10))

    return X_train, Y_train, X_test
