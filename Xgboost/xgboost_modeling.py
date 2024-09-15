import numpy as np
import pandas as pd

# ?~M??~]??~D? ?~H?~_??~X?기 (?~X~H: 'data.csv' ?~L~L?~]?)
data = pd.read_csv('speed_dating_data_1.csv')
columns_to_drop = data.columns[:31]
data = data.drop(columns=columns_to_drop)
data = data.drop(columns=['expected_num_interested_in_me'])


# 결측?~X NaN?~\??~\ ?~L~@체
data.replace(-99, np.nan, inplace=True)



# ?~A 결측?~X ?~R?~]~D ?~O~I?| ?~\??~\ ?~X리
for col in data.columns:
    data[col].fillna(data[col].mean(), inplace=True)




np.random.seed(100)

# ?~M??~]??~D? ?~E~T?~T~L?~A
shuffled_indices = np.random.permutation(len(data))

# ?~E~L?~J??~J? ?~D??~J? ?~A?기 ?~D?~B?
test_set_size = int(len(data) * 0.2)

# ?~]??~M??~J? ?~B~X?~H~D기
test_indices = shuffled_indices[:test_set_size]
train_indices = shuffled_indices[test_set_size:]

# ?~M??~]??~D? ?~B~X?~H~D기
X_train = data.drop('match', axis=1).iloc[train_indices]
X_test = data.drop('match', axis=1).iloc[test_indices]
y_train = data['match'].iloc[train_indices]
y_test = data['match'].iloc[test_indices]




columns_pm = [col for col in data.columns if (col.endswith('_p') | col.endswith('_m'))]
columns_other = [col for col in data.columns if col not in columns_pm]

for col in columns_pm:
    min_val = X_train[col].min()
    max_val = X_train[col].max()
    X_train[col] = (X_train[col] - min_val) / (max_val - min_val)
    X_test[col] = (X_test[col] - min_val) / (max_val - min_val)


import numpy as np
import pandas as pd
from math import e

class Node:
    def __init__(self, x, gradient, hessian, idxs, subsample_cols=0.8, min_leaf=5, min_child_weight=1, depth=10, lambda_=1, gamma=1, eps=0.1):
        self.x, self.gradient, self.hessian = x, gradient, hessian
        self.idxs = idxs
        self.depth = depth
        self.min_leaf = min_leaf
        self.lambda_ = lambda_
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.row_count = len(idxs)
        self.col_count = x.shape[1]
        self.subsample_cols = subsample_cols
        self.eps = eps
        self.column_subsample = np.random.permutation(self.col_count)[:round(self.subsample_cols * self.col_count)]

        self.val = self.compute_gamma(self.gradient[self.idxs], self.hessian[self.idxs])
        self.score = float('-inf')
        self.find_varsplit()

    def compute_gamma(self, gradient, hessian):
        return -np.sum(gradient) / (np.sum(hessian) + self.lambda_)

    def find_varsplit(self):
        for c in self.column_subsample:
            self.find_greedy_split(c)
        if self.is_leaf():
            return
        x = self.split_col()
        lhs = np.nonzero(x <= self.split)[0]
        rhs = np.nonzero(x > self.split)[0]
        self.lhs = Node(self.x, self.gradient, self.hessian, self.idxs[lhs], self.subsample_cols, self.min_leaf, self.min_child_weight, self.depth-1, self.lambda_, self.gamma, self.eps)
        self.rhs = Node(self.x, self.gradient, self.hessian, self.idxs[rhs], self.subsample_cols, self.min_leaf, self.min_child_weight, self.depth-1, self.lambda_, self.gamma, self.eps)

    def find_greedy_split(self, var_idx):
        x = self.x[self.idxs, var_idx]
        for r in range(self.row_count):
            lhs = x <= x[r]
            rhs = x > x[r]
            lhs_indices = np.nonzero(x <= x[r])[0]
            rhs_indices = np.nonzero(x > x[r])[0]
            if(rhs.sum() < self.min_leaf or lhs.sum() < self.min_leaf or self.hessian[lhs_indices].sum() < self.min_child_weight or self.hessian[rhs_indices].sum() < self.min_child_weight):
                continue
            curr_score = self.gain(lhs, rhs)
            if curr_score > self.score:
                self.var_idx = var_idx
                self.score = curr_score
                self.split = x[r]

    def gain(self, lhs, rhs):
        gradient = self.gradient[self.idxs]
        hessian = self.hessian[self.idxs]
        lhs_gradient = gradient[lhs].sum()
        lhs_hessian = hessian[lhs].sum()
        rhs_gradient = gradient[rhs].sum()
        rhs_hessian = hessian[rhs].sum()
        gain = 0.5 * ((lhs_gradient**2 / (lhs_hessian + self.lambda_)) + (rhs_gradient**2 / (rhs_hessian + self.lambda_)) - ((lhs_gradient + rhs_gradient)**2 / (lhs_hessian + rhs_hessian + self.lambda_))) - self.gamma
        return gain

    def split_col(self):
        return self.x[self.idxs, self.var_idx]

    def is_leaf(self):
        return self.score == float('-inf') or self.depth <= 0

    def predict(self, x):
        return np.array([self.predict_row(xi) for xi in x])

    def predict_row(self, xi):
        if self.is_leaf:
            return self.val
        node = self.lhs if xi[self.var_idx] <= self.split else self.rhs
        return node.predict_row(xi)

class XGBoostClassifier:
    def __init__(self):
        self.estimators = []

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def grad(self, preds, labels):
        preds = self.sigmoid(preds)
        return preds - labels

    def hess(self, preds, labels):
        preds = self.sigmoid(preds)
        return preds * (1 - preds)

    def fit(self, X, y, subsample_cols=0.8, min_child_weight=1, depth=5, min_leaf=5, learning_rate=0.4, boosting_rounds=5, lambda_=1.5, gamma=1, eps=0.1, eval_set=None, early_stopping_rounds=None):
        self.X, self.y = X, y
        self.depth = depth
        self.subsample_cols = subsample_cols
        self.eps = eps
        self.min_child_weight = min_child_weight
        self.min_leaf = min_leaf
        self.learning_rate = learning_rate
        self.boosting_rounds = boosting_rounds
        self.lambda_ = lambda_
        self.gamma = gamma
        self.eval_set = eval_set
        self.early_stopping_rounds = early_stopping_rounds

        self.base_pred = np.full((X.shape[0], 1), 1).flatten().astype('float64')
        best_score = float('inf')
        best_round = 0

        for booster in range(self.boosting_rounds):
            Grad = self.grad(self.base_pred, self.y)
            Hess = self.hess(self.base_pred, self.y)
            boosting_tree = XGBoostTree().fit(self.X, Grad, Hess, depth=self.depth, min_leaf=self.min_leaf, lambda_=self.lambda_, gamma=self.gamma, eps=self.eps, min_child_weight=self.min_child_weight, subsample_cols=self.subsample_cols)
            self.base_pred += self.learning_rate * boosting_tree.predict(self.X)
            self.estimators.append(boosting_tree)

            # 조기 종료를 위한 평가
            if self.eval_set:
                eval_X, eval_y = self.eval_set
                eval_pred = self.predict(eval_X)
                eval_score = np.mean((eval_y - eval_pred) ** 2)  # MSE 평가
                if eval_score < best_score:
                    best_score = eval_score
                    best_round = booster
                elif self.early_stopping_rounds and booster - best_round >= self.early_stopping_rounds:
                    print(f"Early stopping at round {booster}")
                    break

    def predict_proba(self, X):
        pred = np.zeros(X.shape[0])
        for estimator in self.estimators:
            pred += self.learning_rate * estimator.predict(X)
        return self.sigmoid(pred)

    def predict(self, X):
        predicted_probas = self.predict_proba(X)
        preds = np.where(predicted_probas > 0.5, 1, 0)
        return preds

class XGBoostTree:

    def fit(self, x, gradient, hessian, subsample_cols = 0.8 , min_leaf = 5, min_child_weight = 1 ,depth = 10, lambda_ = 1, gamma = 1, eps = 0.1):
        self.dtree = Node(x, gradient, hessian, np.array(np.arange(len(x))), subsample_cols, min_leaf, min_child_weight, depth, lambda_, gamma, eps)
        return self

    def predict(self, X):
        return self.dtree.predict(X)





np.random.seed(100)

# 분류 문제 예제
classifier = XGBoostClassifier()
classifier.fit(X_train.values, y_train.values, subsample_cols=0.8, min_child_weight=1, depth=5, min_leaf=5, learning_rate=0.1, boosting_rounds=50, lambda_=1.0, gamma=1, eps=0.1, early_stopping_rounds=10, eval_set=(X_test.values, y_test.values))

# 예측 수행
y_pred_train = classifier.predict(X_train.values)
y_pred_test = classifier.predict(X_test.values)

# 모델 성능 평가
train_accuracy = np.mean(y_train.values == y_pred_train)
test_accuracy = np.mean(y_test.values == y_pred_test)
print(f'Train Accuracy: {train_accuracy}')
print(f'Test Accuracy: {test_accuracy}')
