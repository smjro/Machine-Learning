import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import code
from adalinegd import AdalineGD
from plot_decision_regions import plot_decision_regions

df = pd.read_csv('https://archive.ics.uci.edu/ml/'
                 'machine-learning-databases/iris/iris.data', header=None)

# 1-100行目の目的変数の抽出
y = df.iloc[0:100, 4].values
# Iris-setosaを-1, Iris-virginicaを1に変換
y = np.where(y == 'Iris-setosa', -1, 1)
# 1-100行目の1,3列目を抽出
X = df.iloc[0:100, [0, 2]].values
# 品種setosaのプロット（赤の〇）
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
# 品種virginicaのプロット（青の×）
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='virginica')
# 軸のラベルの設定
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
# 凡例の設定（左上に配置）
plt.legend(loc='upper left')
# 図の表示
plt.show()

# code.InteractiveConsole(globals()).interact()

# データのコピー
X_std = np.copy(X)
# 各列の標準化
X_std[:,0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:,1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()
# 勾配降下法によるADALINEの学習（標準化後、学習率eta=0.01）
ada = AdalineGD(n_iter=15, eta=0.01)
# モデルの適合
ada.fit(X_std, y)
# 境界線のプロット
plot_decision_regions(X_std, y, classifier=ada)
# タイトルの設定
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
# 凡例の設定（左上に配置）
plt.legend(loc='upper left')
# 図の表示
plt.show()
# エポック数とコストの関係を表す折れ線グラフのプロット
plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
# 軸ラベルの設定
plt.xlabel('Epocks')
plt.ylabel('Sum-squared-error')
# 図の表示
plt.show()
