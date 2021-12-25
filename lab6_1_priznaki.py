import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest,chi2,RFE
from sklearn.ensemble import RandomForestClassifier

# Шаг 1

df = pd.read_csv("data/accidental-deaths-in-usa-monthly.csv")

year = []
for i in df['Month']:
    year.append(datetime.datetime.strptime(i, "%Y-%m").strftime("%Y"))

Year = pd.Series(year)
df['Year'] = Year

print('Размер Dataframe:', df.shape)
print(df.head(5))


# Шаг 2

print('\n\nИнформация о Dataframe df.info():')
print(df.info())

# Шаг 3 Сохрание столбеца метки в отдельной переменной и его полное удаление из Dataframe.

print('\n\nstep 3')

label = df["Year"]

# Шаг 4 димонстрация данных

print('\nstep 4')

label.value_counts().plot(kind="bar")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(12,12))
plt.bar(df['Month'], df['Accidental deaths in USA: monthly, 1973 ? 1978'])
plt.xticks(rotation=90)
# рисуем всё сразу
plt.show()



