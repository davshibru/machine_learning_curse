import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest,chi2,RFE
from sklearn.ensemble import RandomForestClassifier

# Шаг 1

df = pd.read_csv("data/Heart_Disease_Prediction.csv")
print('Размер Dataframe:', df.shape)
print(df.head(5))

# Шаг 2

print('\n\nИнформация о Dataframe df.info():')
print(df.info())

# Шаг 3 Сохрание столбеца метки в отдельной переменной и его полное удаление из Dataframe.

print('\n\nstep 3')

label = df["Heart Disease"]
df.drop("Heart Disease", axis=1, inplace=True)
print('\nЗначение метки "Heart Disease":')
print(label.value_counts())


# Шаг 4 димонстрация данных

print('\nstep 4')

label.value_counts().plot(kind="bar")
plt.xticks(rotation=45)
plt.show()

# Шаг 5 измение типа данных на категориальный

print('\nstep 5')

categorical_features = ["Age", "Sex", "Chest pain type", "BP", "Cholesterol", "FBS over 120", "EKG results", "Max HR", "ST depression", "Slope of ST", "Number of vessels fluro", "Thallium"]
df[categorical_features] = df[categorical_features].astype("category")


# Шаг 6 MinMaxScaler

print('\nstep 6')

continuous_features = set(df.columns) - set(categorical_features)
scaler = MinMaxScaler()
df_norm = df.copy()
df_norm[list(continuous_features)] = scaler.fit_transform(df[list(continuous_features)])

# Шаг 7 Отбор признаков с помощью \chi^2χ2

print('\nstep7')

X_new = SelectKBest(k=5, score_func=chi2).fit_transform(df_norm, label)

# Шаг 8 Отбор признаков с использованием рекурсивного исключения признаков

print('\nstep 8')

rfe = RFE(estimator=RandomForestClassifier(), n_features_to_select=5)
X_new = rfe.fit_transform(df_norm, label)


# Шаг 9 Отбор признаков с использованием случайного леса

print('\nstep9')

clf = RandomForestClassifier()
clf.fit(df_norm, label)
plt.figure(figsize=(12,12))
plt.bar(df_norm.columns, clf.feature_importances_)
plt.xticks(rotation=45)

plt.show()

