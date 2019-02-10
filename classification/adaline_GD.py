import numpy as np


class AdalineGD(object):
    """ ADAptive Linear NEron分類器
    
    パラメータ
    -----------------------------
    eta : float
        学習率（0.0より大きく1.0以下の値）
    n_iter : int
        トレーニングデータのトレーニング回数

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
        self.cost_ = []

        for _ in range(self.n_iter):  # トレーニング回数分トレーニングデータを反復
            # 活性化関数の出力を計算
            output = self.net_input(x)
            # 誤差 y^i - φ(z^i) の計算
            errors = y - output
            # 重み w_1, ..., w_mの更新
            # Δw_j = ηΣ_i(y^i - φ(z^i))x_j^i (j = 1, ..., m)
            self.w_[1:] += self.eta * x.T.dot(errors)
            # 重み w_0の更新 : Δw_0 = ηΣ_i(y^i - φ(z^i))
            self.w_[0] += self.eta * errors.sum()
            # コスト関数の計算 J(w) = 1/2 Σ_i(y^i - φ(z^i))^2
            cost = (errors**2).sum() / 2.0
            # コストの格納
            self.cost_.append(cost)
        return self

    def net_input(self, x):
        """ 総入力を計算 """
        return np.dot(x, self.w_[1:]) + self.w_[0]

    def activation(self, x):
        """ 線形活性化関数の出力を計算 """
        return self.net_input(x)

    def predict(self, x):
        """ 1ステップ後のクラスラベルを返す """
        return np.where(self.activation(x) >= 0.0, 1, -1)
