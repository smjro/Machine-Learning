import numpy as np


class Perceptron(object):
    """ パーセプトロン
    
    パラメータ
    -----------------------------
    eta : float
        学習率（0.0より大きく1.0以下の値）
    n_iter : int
        トレーニングデータのトレーニング回数
    random_state : int
        重みを初期化するための乱数シード

    属性
    -----------------------------
    w_ : 1d-array
        適合後の重み
    errors_ : list
        各エポックでの誤分類数
    """

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, x, y):
        """ トレーニングデータに適合させる

        パラメータ
        -----------------------------
        X : {array-like}, shape = [n_samples, n_features]
            トレーニングデータ
        y : array-like, shape = [n_samples]
            目的変数

        戻り値
        -----------------------------
        self : object
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + x.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):  # トレーニング回数分トレーニングデータを反復
            errors = 0
            for xi, target in zip(x, y):  # 各サンプルで重みを更新
                # 重み w_1, ..., w_mの更新
                # Δw_j = η(y^i - yhat^i)x_j^i (j = 1, ..., m)
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                # 重み w_0の更新 : Δw_0 = η(y^i - yhat^i)
                self.w_[0] += self.eta * update
                # 重みの更新が 0 でない場合は誤分類としてカウント
                errors += int(update != 0.0)
            # 反復回数ごとの誤差を格納
            self.errors_.append(errors)
        return self

    def net_input(self, x):
        """ 総入力を計算 """
        return np.dot(x, self.w_[1:]) + self.w_[0]

    def predict(self, x):
        """ 1ステップ後のクラスラベルを返す """
        return np.where(self.net_input(x) >= 0.0, 1, -1)
