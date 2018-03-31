""" Class of Ridge regressor. """

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import Ridge


class RidgeRegressor:
    """ A Ridge Regressor that takes train and test data as input, output predict value.
    
    Attributes:
        train: training matrix
        test: test matrix
        cv_ratio: ratio of cv data over all training data, default at 0.2
    """

    def __init__(self, train, test, cv_ratio=0.2, verbose=True):
        self.train = train
        self.test = test
        self.cv_ratio = cv_ratio
        self.verbose = verbose

    def predict(self):
        """ Predict test dataset with search best alpha value in train/cv dataset.
        """
        train_cv_x = self.train[:, :-1]
        train_cv_y = self.train[:, -1]
        test_x = self.test[:, :-1]
        test_y = self.test[:, -1]

        train_x, cv_x, train_y, cv_y = train_test_split(train_cv_x, train_cv_y, test_size=self.cv_ratio)

        if self.verbose:
            print('\n>>> Shape of train matrix: {0} x {1}'.format(*train_x.shape))
            print('>>> Shape of cv matrix: {0} x {1}'.format(*cv_x.shape))
            print('>>> Shape of test matrix: {0} x {1}'.format(*test_x.shape))

        # grid search best alpha value over -5 to 5 in log space
        alpha_array = [10 ** t for t in range(-5, 5)]
        cv_mae = []
        for alpha in alpha_array:
            predictor = Ridge(alpha=alpha)
            predictor.fit(train_x, train_y)
            cv_yhat = predictor.predict(cv_x)
            mae = mean_absolute_error(cv_y, cv_yhat)
            if self.verbose:
                print('>>> CV phase, MAE at alpha {0}: {1:.4f}'.format(alpha, mae))
            cv_mae.append(mae)

        # build the best predictor
        best_alpha_idx = np.argmin(np.array(cv_mae))
        best_alpha = alpha_array[best_alpha_idx]
        if self.verbose:
            print('>>> Best hyper parameter alpha: {0}'.format(best_alpha))
        best_predictor = Ridge(alpha=best_alpha)
        best_predictor.fit(train_cv_x, train_cv_y)

        # predict test dataset
        test_yhat = best_predictor.predict(test_x)
        if self.verbose:
            print('>>> Predict {0} videos in test dataset'.format(len(test_yhat)))
            print('>>> Ridge model: MAE of test dataset: {0:.4f}'.format(mean_absolute_error(test_y, test_yhat)))
            print('>>> Ridge model: R2 of test dataset: {0:.4f}'.format(r2_score(test_y, test_yhat)))
        return test_yhat

    def predict_from_sparse(self, vectorize_train_func, vectorize_test_func):
        """ Predict test sparse dataset with search best alpha value in sparse train/cv dataset.
        """
        train_matrix, cv_matrix = train_test_split(self.train, test_size=self.cv_ratio)

        if self.verbose:
            print('>>> Length of train matrix: {0}'.format(len(train_matrix)))
            print('>>> Length of cv matrix: {0}'.format(len(cv_matrix)))
            print('>>> Length of test matrix: {0}\n'.format(len(self.test)))

        # generate train dataset on the fly
        train_sparse_x, train_y, train_topics = vectorize_train_func(train_matrix)
        cv_sparse_x, cv_y, _ = vectorize_test_func(cv_matrix, train_topics)

        # grid search over alpha in ridge regressor
        alpha_array = [10 ** t for t in range(-5, 5)]
        cv_mae = []
        for alpha in alpha_array:
            predictor = Ridge(alpha=alpha)
            predictor.fit(train_sparse_x, train_y)
            cv_yhat = predictor.predict(cv_sparse_x)
            mae = mean_absolute_error(cv_y, cv_yhat)
            if self.verbose:
                print('>>> Sparse CV phase, MAE: {0} with alpha value: {1}'.format(mae, alpha))
            cv_mae.append(mae)

        # build the best estimator
        best_alpha_idx = np.argmin(np.array(cv_mae))
        best_alpha = alpha_array[best_alpha_idx]
        if self.verbose:
            print('>>> best hyper parameter alpha: {0}'.format(best_alpha))
        best_predictor = Ridge(alpha=best_alpha)
        train_sparse_x, train_y, train_topics = vectorize_train_func(self.train)
        best_predictor.fit(train_sparse_x, train_y)

        # build test dataset on the fly
        test_sparse_x, test_y, test_vids = vectorize_test_func(self.test, train_topics)
        test_yhat = best_predictor.predict(test_sparse_x)
        if self.verbose:
            print('>>> Predict {0} videos in test dataset'.format(len(test_yhat)))
            print('>>> Ridge sparse model: MAE of test dataset: {0:.4f}'.format(mean_absolute_error(test_y, test_yhat)))
            print('>>> Ridge sparse model: R2 of test dataset: {0:.4f}'.format(r2_score(test_y, test_yhat)))
        return test_yhat, test_vids
