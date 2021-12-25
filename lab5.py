from sklearn.datasets import load_wine

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn import mixture

# 1 обучение с учителем, гауссов нивный байесов классификатор
print('\n1 обучение с учителем, гауссов нивный байесов классификатор')

# Шаг 1 подключение гауссова нивного байесовского классификатора
from sklearn.naive_bayes import GaussianNB

wine = load_wine()

# Шаг 2 извлечение матрицы признаков
X_wine, Y_wine = load_wine(return_X_y=True)
print(X_wine)
print(Y_wine)
# Шаг 3 создание тренировочного и контрольного набора данных
Xtrain, Xtest, ytrain, ytest = train_test_split(X_wine, Y_wine, random_state=1)

model = GaussianNB()

# Шаг 4 обучение модели на данных
model.fit(Xtrain, ytrain)

# Шаг 5 предсказываем значения для новых данных
y_model = model.predict(Xtest)

# Шаг 6 подключение утилиты для проверки полученных данных
from sklearn.metrics import accuracy_score

# Шаг 7 вывод данных
print(f'\nТочность предсказания - {accuracy_score(ytest, y_model)}')

mat = confusion_matrix(ytest, y_model)
sns.heatmap(mat, square = True, annot = True, cbar = False)
plt.xlabel('predicted value')
plt.ylabel('true value')
plt.show()

# 2	обучение без учителя, кластеризация

print('\n2 обучение без учителя, кластеризация')

# Шаг 1 создание экземпляра модели
model = mixture.GaussianMixture(n_components=3, covariance_type='full')

# Шаг 2 обучение модели на данных
model.fit(X_wine)

# Шаг 3 графическое отображение данных
plt.scatter(X_wine[:, 0], X_wine[:, -1], c=model.predict(X_wine), alpha=0.5)
plt.show()



# 3 обучения без учителя, понижение размероности

print('\n3 обучения без учителя, понижение размероности')

# Шаг 1 подключение библиотеки
from sklearn.decomposition import PCA
import pandas as pd

# Шаг 2 создание экземпляра модели
model = PCA(n_components = 2)

# Шаг 3 обучение модели на данных
model.fit(X_wine)

# Шаг 4 преоброзование данных в двумерные
X_2D  = model.transform(X_wine)

_2D = pd.DataFrame()

_2D['PCA1'] = X_2D[:, 0]
_2D['PCA2'] = X_2D[:, 1]


# Шаг 5 графическое отображение данных

sns.lmplot(x = 'PCA1', y = 'PCA2', data = _2D,  fit_reg = False)
plt.show()

# 4	линейная регрессия

print('\n4 линейная регрессия')

# Шаг 1 импорт библиотек
from sklearn.linear_model import LinearRegression
import numpy as np

# Шаг 2 создание экземпляра модели

model = LinearRegression()


model.fit(X_wine, Y_wine)

# r_sq = model.score(X_wine, Y_wine)
# print(f'coefficient of determination - {r_sq}')
predict_data = model.predict(Xtest)
#
wine_df = pd.DataFrame(data= wine.data, columns= wine.feature_names)
target_df = pd.DataFrame(data= wine.target, columns= ['species'])
#

xfit = model.predict(Xtest)
print(xfit)

Xfit = xfit[:, np.newaxis]

# yfit = model.predict(Xfit)

# plt.scatter(X_wine, Y_wine)
plt.scatter(X_wine[:, 0], X_wine[:, -1])

mat = confusion_matrix(ytest, predict_data)

plt.show()

# model = LinearRegression()
# model.fit(iris_df, target_df)
#
# # Concatenate the DataFrames
# iris_df = pd.concat([iris_df, target_df], axis= 1)
#
# iris_df.describe()
# iris_df.info()
# plt.plot(iris_df, target_df)
# plt.show()
