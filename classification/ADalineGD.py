import numpy as np

class AdalineGD(object):
    """ADaptive LInear NEuron分類器

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
    errors : リスト
        各エポックでの誤分類数

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
        # バイアス分の要素を追加
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):    # トレーニング回数分トレーニングデータを反復
            # 活性化関数の出力の計算　φ(w^T x) = w^T x
            output = self.net_input(X)
            # 誤差 y - φ(z)の計算
            errors = (y - output)
            # w1, ... , wmの更新
            # Δw_j = ηΣ(y_i - φ(z_i))x_j (j = 1, ... , m)
            self.w_[1:] += self.eta * X.T.dot(errors)
            # w0(バイアス) の更新 Δw0 = ηΣ(y_i - φ(z_i))
            self.w_[0] += self.eta * errors.sum()
            # コスト関数の計算 J(w) = 1/2 Σ(y_i - φ(z_i))^2
            cost = (errors**2).sum() / 2.0
            # コストの格納
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """ 総入力を計算 """
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """ 線形活性化関数の出力を計算 """
        return self.net_input(X)

    def predict(self, X):
        """ 1ステップ後のクラスラベルを返す """
        return np.where(self.activation(X) >= 0.0, 1, -1)
            
