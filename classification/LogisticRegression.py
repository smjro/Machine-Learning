import numpy as np

class LogisticRegression(object):
    """ ロジスティック回帰の分類器
    
    パラメータ
    ----------
    eta : float
        学習率（0.0より大きく1.0以下の値）
    n_iter : int
        トレーニングデータのトレーニング回数
    
    属性
    ----------
    w_ : 1次元配列
        適合後の重み
    cost_ : リスト
        各エポックでのコスト

    """
    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """ トレーニングデータに適合させる

        パラメータ
        ----------
        X : {配列のようなデータ構造}, shape = [n_samples, n_features]
            トレーニングデータ
            n_samplesはサンプルの個数、n_featureは特徴量の個数
        y : 配列のようなデータ構造, shape = [n_samples]
            目的変数

        戻り値
        ----------
        self : object

        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter): # トレーニング回数分トレーニングデータを反復
            y_val = self.activation(X)
            errors = (y - y_val);
            neg_grad = X.T.dot(errors)
            self.w_[1:] += self.eta * neg_grad
            self.w_[0] += self.eta * errors.sum()
            # 反復回数ごとのコストを格納
            self.cost_.append(self._logit_cost(y, self.activation(X)))
        return self

    def _logit_cost(self, y, y_val):
        logit = -y.dot(np.log(y_val)) - ((1-y).dot(np.log(1 - y_val)))
        return logit

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def _softmax(self, z):
        return np.exp(z) / np.sum(np.exp(z))

    def net_input(self, X):
        """ 総入力を計算 """
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """ 線形活性化関数の出力を計算 """
        z = self.net_input(X)
        return self._softmax(z)
        #return self._sigmoid(z)

    def predict_proba(self, X):
        """ 予測確率を返す """
        return activation(X)

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)

