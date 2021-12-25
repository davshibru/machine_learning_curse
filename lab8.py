import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import fetch_20newsgroups


from sklearn import mixture


df = pd.read_csv("data/Fremont_Bridge_Bicycle_Counter.csv")
df_hours = pd.read_csv("data/Fremont_Bridge_Bicycle_Counter.csv")

# Шаг 1 создание колонки hours

hours = []
for i in df_hours['Date']:

     hours.append(datetime.datetime.strptime(i, "%m/%d/%Y %H:%M:%S %p").strftime("%H"))


Hours = pd.Series(hours)
df_hours['Hours'] = Hours

print('Размер Dataframe:', df_hours.shape)



print(df_hours.head(30))

print('\n\nИнформация о Dataframe df.info():')
print(df.info())

# Шаг 2 демонстрация граффика

print(df.head(5))
df.plot(figsize=(20, 4))
plt.ylabel("Количество велосепедистов")
# plt.show()





df_hours['Date'] = pd.to_datetime(df['Date'])


df_hours = df_hours.set_index('Date')
df_hours = df_hours.reindex(columns=['Fremont Bridge East Sidewalk', 'Fremont Bridge West Sidewalk', 'Hours'])
indep_cols = ['Fremont Bridge East Sidewalk', 'Fremont Bridge West Sidewalk', 'Hours']
print('dddddddddddddddddddddddddddddddd')
print(f'-----------\n{df_hours.head(10)}')

x = df[indep_cols]
y = df['Fremont Bridge Total']

#
# # 3
# # извлечение матрицы признаков
# X_iris = df_hours.drop('Date', axis = 1)
#
# # извлечение целевого массива
# Y_iris = df_hours['Date']
#
# # 2 создание экземпляра модели
#
# model = mixture.GaussianMixture(n_components=3, covariance_type='full')
#
# # 4 обучение модели на данных
# model.fit(X_iris)
#
# # определяем метки кластеров
# y_gmm = model.predict(X_iris)
#
# print(y_gmm)