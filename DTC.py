import numpy as np

"""CART - Classification And Regression Trees In the case of Classification Trees, CART algorithm uses a metric 
called Gini Impurity to create decision points for classification tasks. Gini Impurity gives an idea of how fine a 
split is. In the case of Regression Trees, CART algorithm looks for splits that minimize the Least Square Deviation (
LSD), choosing the partitions that minimize the result over all possible options. The LSD (sometimes referred as 
“variance reduction”) metric minimizes the sum of the squared distances (or deviations) between the observed values 
and the predicted values. """

class DTC:
    def __init__(self):
        self.root = None
        self.max_depth = 0
        self.X = None
        self.y = None
        self.n_classes = None

    @staticmethod
    def gini_index(groups, y):
        n_instances = len(groups[0]) + len(groups[1])
        gini = 0.0
        for indexes in groups:
            size = len(indexes)
            if size == 0:
                continue
            score = 0.0
            for class_val in np.unique(y):
                p = (y[indexes] == class_val).sum() / size
                score += p * p
            gini += (1 - score) * (size / n_instances)
        return gini

    def get_split(self, X, y):
        b_index, b_value, b_score, b_groups = float('inf'), float('inf'), float('inf'), None
        for col_ind in range(X.shape[1]):
            for val in np.unique(X[:, col_ind]):

                left_index = np.reshape(np.argwhere(X[:, col_ind] < val), (-1,))
                right_index = np.reshape(np.argwhere(X[:, col_ind] >= val), (-1,))

                gini = self.gini_index((left_index, right_index), y)

                if gini < b_score:
                    b_index, b_value, b_score, b_groups = col_ind, val, gini, (left_index, right_index)

        return {'index': b_index, 'value': b_value, 'groups': b_groups}

    def to_terminal(self, classes):
        cls, cnt = np.unique(classes, return_counts=True)
        probs = np.zeros(self.n_classes)
        for cl, cn in zip(cls, cnt):
            probs[cl] = cn / sum(cnt)
        return cls[np.argmax(cnt)], probs

    def split(self, node, X, y, max_depth, min_samples_split, depth):
        self.max_depth = max(depth, self.max_depth)
        left, right = node.pop('groups')

        if len(left) == 0 or len(right) == 0:
            node['left'] = node['right'] = self.to_terminal(y[np.append(left, right)])
            return

        if depth >= max_depth:
            node['left'], node['right'] = self.to_terminal(y[left]), self.to_terminal(y[right])
            return

        if len(left) <= min_samples_split:
            node['left'] = self.to_terminal(y[left])
        else:
            node['left'] = self.get_split(X[left], y[left])
            self.split(node['left'], X[left], y[left], max_depth, min_samples_split, depth + 1)

        if len(right) <= min_samples_split:
            node['right'] = self.to_terminal(y[right])
        else:
            node['right'] = self.get_split(X[right], y[right])
            self.split(node['right'], X[right], y[right], max_depth, min_samples_split, depth + 1)

    def fit(self, X, y, max_depth=None, min_samples_split=2):
        self.X, self.y, max_depth = X, y, float('inf') if max_depth == None else max_depth
        self.n_classes = len(np.unique(y))
        self.root = self.get_split(X, y)
        self.split(self.root, X, y, max_depth, min_samples_split, 1)

    def predict(self, rows):
        return np.array([self.predict_row(row, self.root)[0] for row in rows])

    def predict_proba(self, rows):
        return np.array([self.predict_row(row, self.root)[1] for row in rows])

    def predict_row(self, row, node):
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return self.predict_row(row, node['left'])
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self.predict_row(row, node['right'])
            else:
                return node['right']

    def score(self, X, y):
        return (y == self.predict(X)).sum() / len(y)

    @property
    def depth(self):
        return self.max_depth

    @property
    def tree_(self):
        return self.root
