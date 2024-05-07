import numpy as np

from scipy.stats import uniform, randint

from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split

import xgboost as xgb
import matplotlib.pyplot as plt
from graphviz import Source

def display_scores(scores):
    print("Scores: {0}\nMean: {1:.3f}\nStd: {2:.3f}".format(scores, np.mean(scores), np.std(scores)))


if __name__ == '__main__':
    diabetes = load_diabetes()

    X = diabetes.data
    # y = diabetes.target
    y = diabetes.data
    xgb_model = xgb.XGBRegressor(objective="reg:linear", random_state=42)

    xgb_model.fit(X, y)

    y_pred = xgb_model.predict(X)

    mse = mean_squared_error(y, y_pred)

    print(np.sqrt(mse))

    xgb.plot_importance(xgb_model)
    # plt.show()
    xgb.to_graphviz(xgb_model, num_trees=xgb_model.best_iteration)
    plt.show()

    print('end')

