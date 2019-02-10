import numpy as np
from numpy.random import seed


class AdalineSGD(object):
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
    shuffle : bool （デフォルト : True）
        循環を回避するために各エポックでトレーニングデータをシャッフル
    random_state : int （デフォルト : None）
        シャッフルに使用するランダムステートを設定し、重みを初期化

    """
    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        # 学習率の初期化
        self.eta = eta
        # トレーニング回数の初期化
        self.n_iter = n_iter
        # 重みの初期化フラグはFalseに設定
        self.w_initialized = False
        # 各エポックでトレーニングデータをシャッフルするかどうかのフラグを初期化
        self.shuffle = shuffle
        # 乱数のシードを設定
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
        # 重みベクトルの生成
        self._initialize_weights(x.shape[1])
        # コストを格納するリストの生成
        self.cost_ = []
        # トレーニング回数分トレーニングデータを反復
        for _ in range(self.n_iter):
            # 指定された場合はトレーニングデータをシャッフル
            if self.shuffle:
                x, y = self._shuffle(x, y)
            # 各サンプルのコストを格納するリストを生成
            cost = []
            # 各サンプルに対する計算
            for xi, target in zip(x, y):
                # 特徴量 xi と目的変数 y を用いた重みの更新とコストの計算
                cost.append(self._update_weights(xi, target))
            # サンプルの平均コストの計算
            avg_cost = sum(cost) / len(y)
            # 平均コストを格納
            self.cost_.append(avg_cost)
        return self

    def partial_fit(self, x, y):
        """ 重みを再初期化することなくトレーニングデータに適合させる """
        # 初期化されていない場合は初期化を実行
        if not self.w_initialized:
            self._initialize_weights(x.shape[1])
        # 目的変数 y の要素数が2以上の場合は
        # 各サンプルの特徴量 xi と目的変数 target で重みを更新
        if y.ravel().shape[0] > 1:
            for xi, target in zip(x, y):
                self._update_weights(xi, target)
        # 目的変数 y の要素数が1の場合は
        # サンプル全体の特徴量 X と目的変数 y で重みを更新
        else:
            self._update_weights(x, y)
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

    def _initialize_weights(self, m):
        """ 重みを小さな乱数に初期化 """
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=1 + m)
        self.w_initialized = True

    def _shuffle(self, x, y):
        """ トレーニングデータをシャッフル """
        r = self.rgen.permutation(len(y))
        return x[r], y[r]

    def _update_weights(self, x, y):
        """ ADALINEの学習規則を用いて重みを更新 """
        # 活性化関数の出力を計算
        output = self.net_input(x)
        # 誤差 y - φ(z) の計算
        error = y - output
        # 重み w_1, ..., w_mの更新
        # Δw_j = η(y - φ(z))x_j (j = 1, ..., m)
        self.w_[1:] += self.eta * x.dot(error)
        # 重み w_0の更新 : Δw_0 = η(y - φ(z))
        self.w_[0] += self.eta * error
        # コスト関数の計算 J(w) = 1/2 (y - φ(z))^2
        cost = 0.5 * error**2
        return cost
