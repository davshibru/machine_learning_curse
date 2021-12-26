# Шаг 1 импорт библиотек

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import fetch_20newsgroups
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.model_selection  import train_test_split

from sklearn import mixture

# Шаг 2 получение дата сета

df = pd.read_csv("data/Fremont_Bridge_Bicycle_Counter.csv")

# Шаг 3 демонстрация граффика


df.plot(figsize=(20, 4))
plt.ylabel("Количество велосепедистов")
plt.show()

# Шаг 4 уменьшение количество строк в df


r'''

Так как при размере строк больше 12200 при попытке использовать предикт ленейной регресии выдается ошибка ниже. Поэтому было решено уменьшить количество строк до 12000

Ошибка:

Traceback (most recent call last):
  File "C:/Users/Gadget/PycharmProjects/iris/lab8.py", line 104, in <module>
    model.fit(Xtrain, ytrain)                         # 3. fit model to train data
  File "C:\Users\Gadget\PycharmProjects\iris\venv\lib\site-packages\sklearn\linear_model\_base.py", line 518, in fit
    X, y = self._validate_data(X, y, accept_sparse=accept_sparse,
  File "C:\Users\Gadget\PycharmProjects\iris\venv\lib\site-packages\sklearn\base.py", line 433, in _validate_data
    X, y = check_X_y(X, y, **check_params)
  File "C:\Users\Gadget\PycharmProjects\iris\venv\lib\site-packages\sklearn\utils\validation.py", line 63, in inner_f
    return f(*args, **kwargs)
  File "C:\Users\Gadget\PycharmProjects\iris\venv\lib\site-packages\sklearn\utils\validation.py", line 880, in check_X_y
    y = check_array(y, accept_sparse='csr', force_all_finite=True,
  File "C:\Users\Gadget\PycharmProjects\iris\venv\lib\site-packages\sklearn\utils\validation.py", line 63, in inner_f
    return f(*args, **kwargs)
  File "C:\Users\Gadget\PycharmProjects\iris\venv\lib\site-packages\sklearn\utils\validation.py", line 720, in check_array
    _assert_all_finite(array,
  File "C:\Users\Gadget\PycharmProjects\iris\venv\lib\site-packages\sklearn\utils\validation.py", line 103, in _assert_all_finite
    raise ValueError(
ValueError: Input contains NaN, infinity or a value too large for dtype('float64').

'''

df = df[:12250]


# Шаг 5 перевод колонки "Date" из типа данных object в тип данных datetime64

df['Date'] = pd.to_datetime(df['Date'])


# Шаг 6 создание колонки Hours и добавление ее в переменную df

hours = []
for i in df['Date']:

     hours.append(datetime.datetime.strptime(str(i), "%Y-%m-%d %H:%M:%S").strftime("%H"))


Hours = pd.Series(hours)
df['Hours'] = Hours

df['Hours'] = df['Hours'].astype("float64")

# Шаг 7 установка индекса

df = df.set_index('Date')


# Шаг 8 создание колонок которые содержат информацию о дне недели

df = (
     df.
     assign(
          day_of_week=lambda _df: _df.index.dayofweek
     )
          .pipe(pd.get_dummies, columns=['day_of_week'])
          .rename(
          columns={
               'day_of_week_0': 'Mon',
               'day_of_week_1': 'Tue',
               'day_of_week_2': 'Wed',
               'day_of_week_3': 'Thu',
               'day_of_week_4': 'Fri',
               'day_of_week_5': 'Sat',
               'day_of_week_6': 'Sun'
          }
     )
)


# Шаг 9 создание колонки которая показывает евляется ли день праздником

cal = USFederalHolidayCalendar()
holidays = cal.holidays('2012', '2020')
df = df.join(pd.Series(1, index=holidays, name='holiday'))
df['holiday'].fillna(0, inplace=True)


# Шаг 10 разделение данных на “атрибуты” и “метки”

column_names = ['Hours', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun', 'holiday']
X = df[column_names]
y = df['Fremont Bridge Total']


# Шаг 11 разделение данных на обучающие и тестовые наборы.

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=1)


# Шаг 12 создание и подготовка в модели

from sklearn.linear_model import LinearRegression

model = LinearRegression(fit_intercept=False)
model.fit(Xtrain, ytrain)
y_model = model.predict(Xtest)

# Шаг 13 проверка предсказания

from sklearn.metrics import r2_score

s = r2_score(ytest, y_model)

print(s)


# Шаг 14 подговка собственных данных

ddd = pd.DataFrame([['2022-12-25', 12, 0, 0, 0, 0, 0, 1, 0, 0.0]], columns=['Date', 'Hours', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun', 'holiday'])

ddd = ddd.set_index('Date')


print(ddd[:] )
print(model.predict(ddd[:] ))





# прототип

# df = pd.read_csv("data/Fremont_Bridge_Bicycle_Counter.csv")
# df_hours = pd.read_csv("data/Fremont_Bridge_Bicycle_Counter.csv")
#
# # Шаг 1 создание колонки hours
#
#
# hours = []
# for i in df_hours['Date']:
#
#      hours.append(datetime.datetime.strptime(i, "%m/%d/%Y %H:%M:%S %p").strftime("%H"))
#
#
# Hours = pd.Series(hours)
# df_hours['Hours'] = Hours
#
# print('Размер Dataframe:', df_hours.shape)
#
#
#
# print(df_hours.head(30))
#
# print('\n\nИнформация о Dataframe df.info():')
# print(df.info())
#
# # Шаг 2 демонстрация граффика
#
# print(df.head(5))
# df.plot(figsize=(20, 4))
# plt.ylabel("Количество велосепедистов")
# # plt.show()
#
#
#
#
#
# df_hours['Date'] = pd.to_datetime(df['Date'])
#
#
# df_hours = df_hours.set_index('Date')
# df_hours = df_hours.reindex(columns=['Fremont Bridge East Sidewalk', 'Fremont Bridge West Sidewalk', 'Hours'])
# indep_cols = ['Fremont Bridge East Sidewalk', 'Fremont Bridge West Sidewalk', 'Hours']
# print('dddddddddddddddddddddddddddddddd')
# print(f'-----------\n{df_hours.head(10)}')
#
# x = df[indep_cols]
# y = df['Fremont Bridge Total']
#
# #
# # # 3
# # # извлечение матрицы признаков
# # X_iris = df_hours.drop('Date', axis = 1)
# #
# # # извлечение целевого массива
# # Y_iris = df_hours['Date']
# #
# # # 2 создание экземпляра модели
# #
# # model = mixture.GaussianMixture(n_components=3, covariance_type='full')
# #
# # # 4 обучение модели на данных
# # model.fit(X_iris)
# #
# # # определяем метки кластеров
# # y_gmm = model.predict(X_iris)
# #
# # print(y_gmm)
#
#
#
#


#-------------------------------------------------------------------------------------------------------------------

# Определение по дням

# print('xxxxxxxx')
# Xtest[836:] = ddd
#
# print(Xtest[836:])
#
# print(model.predict(Xtest[836:]))

#
# df = pd.read_csv("data/Fremont_Bridge_Bicycle_Counter.csv")
#
#
#
#
# df['Date'] = pd.to_datetime(df['Date'])
# df = df.set_index('Date')
#
# daily = (
#     df
#     .resample('d')
#     .sum()
#     .loc[:, ['Fremont Bridge Total']]
#     .rename(
#         columns={
#             'Fremont Bridge Total': 'Total'
#         }
#     )
# )
# print(daily.shape)
# print(daily.head())
# #
# # monthly = (
# #     df
# #     .resample('m')
# #     .sum()
# # ).plot(figsize = (15,5))
# #
# # by_weekday = (
# #     df
# #     .groupby(df.index.dayofweek)
# #     .mean()
# # )
# # by_weekday.index = ['Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun']
# # by_weekday.plot(style=[':', '--', '-']);
# # plt.show()
#
# daily = (
#     daily
#     .assign(
#         day_of_week=lambda _df: _df.index.dayofweek
#     )
#     .pipe(pd.get_dummies, columns=['day_of_week'])
#     .rename(
#         columns={
#             'day_of_week_0': 'Mon',
#             'day_of_week_1': 'Tue',
#             'day_of_week_2': 'Wed',
#             'day_of_week_3': 'Thu',
#             'day_of_week_4': 'Fri',
#             'day_of_week_5': 'Sat',
#             'day_of_week_6': 'Sun'
#         }
#     )
# )
#
# df = (
#      df.
#      assign(
#           day_of_week=lambda _df: _df.index.dayofweek
#      )
#           .pipe(pd.get_dummies, columns=['day_of_week'])
#           .rename(
#           columns={
#                'day_of_week_0': 'Mon',
#                'day_of_week_1': 'Tue',
#                'day_of_week_2': 'Wed',
#                'day_of_week_3': 'Thu',
#                'day_of_week_4': 'Fri',
#                'day_of_week_5': 'Sat',
#                'day_of_week_6': 'Sun'
#           }
#      )
# )
#
# print('167')
# print(df[['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']].head(30))
# print(df.shape)
# print(daily.shape)
#
#
# from pandas.tseries.holiday import USFederalHolidayCalendar
#
# cal = USFederalHolidayCalendar()
# holidays = cal.holidays('2012', '2020')
# daily = daily.join(pd.Series(1, index=holidays, name='holiday'))
# daily['holiday'].fillna(0, inplace=True)
#
# print(
#     daily
#     .loc[daily.holiday == 1]
#     .reset_index()
#     .sort_values(by= "Date")
#     .tail(10)
# )
#
# print(daily.shape)
# print('-------------')
#
# print(daily.head())
#
# column_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun', 'holiday']
# X = daily[column_names]
# y = daily['Total']
#
#
# from sklearn.model_selection  import train_test_split
# Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=1)
#
# from sklearn.linear_model import LinearRegression # 1. choose model class
# model = LinearRegression(fit_intercept=False)     # 2. instantiate model
# model.fit(Xtrain, ytrain)                         # 3. fit model to train data
# y_model = model.predict(Xtest)                    # 4. predict on new test data
#
# from sklearn.metrics import r2_score
#
#
#
#
# # print('\n\n\n')
# #
# # for i in Xtest:
# #      print(i)
# #
# # print('\n\n\n')
# #
# # print('\n\n\n')
# #
# # print('\n\n\n')
# #
# # for i in y_model:
# #      print(i)
# #
# # print('\n\n\n')
# s = r2_score(ytest, y_model)
#
# print(s)
#
# from sklearn.model_selection  import cross_validate
# cv = cross_validate(model, X, y, cv=10, return_train_score=True)
# cv_df = pd.DataFrame({"train_score": cv["train_score"], "test_score": cv["test_score"]})
# print(cv_df)
#
#
#
# ddd = pd.DataFrame([['2022-12-25', 0, 0, 0, 0, 0, 1, 0, 0.0]], columns=['Date', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun', 'holiday'])
# # df['Date'] = pd.to_datetime(df['Date'])
# ddd = ddd.set_index('Date')
#
#
# print(ddd)
# Xtest.append(ddd)
#
#
#
# print(model.predict(Xtest[836:]))
# print('xxxxxxxx')
# Xtest[836:] = ddd
#
# print(Xtest[836:])
#
# print(model.predict(Xtest[836:]))
