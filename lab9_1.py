import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns;

sns.set()

from sklearn.svm import SVC
from sklearn.decomposition import PCA as RandomizedPCA
from sklearn.pipeline import make_pipeline

# библиотека для загрузки колекции лиц
from sklearn.datasets import fetch_lfw_people

# библиотека для  разбиения  данных на группы (тренировка и контроль)
from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

faces = fetch_lfw_people(min_faces_per_person=60)
print(faces.target_names)
print(faces.images.shape)
# 1348 - количество изображений
# 62 х 47  - размер одного изображения
# 4 байта - размер одноо пикселя

# вывод нескольких лиц на экран
fig, ax = plt.subplots(3, 5)
for i, axi in enumerate(ax.flat):
    axi.imshow(faces.images[i], cmap='bone')
    axi.set(xticks=[], yticks=[], xlabel=faces.target_names[faces.target[i]])
plt.show()

# построение класификатора "метод главных компонент"
# выделение 150 ключевых признаков
pca = RandomizedPCA(n_components=150, whiten=True, random_state=42)
svc = SVC(kernel='rbf', class_weight='balanced')
model = make_pipeline(pca, svc)

# разбиение данных на тренировочные и контрольные
Xtrain, Xtest, ytrain, ytest = train_test_split(faces.data, faces.target, random_state=42)

param_grid = {'svc__C': [1, 5, 10, 50], 'svc__gamma': [0.0001, 0.0005, 0.001, 0.005]}
grid = GridSearchCV(model, param_grid)
grid.fit(Xtrain, ytrain)

print(grid.best_params_)

# построение модели для тестовых данных по оптимальным значениям коэф.
model = grid.best_estimator_
yfit = model.predict(Xtest)

# визуализация части полученых результатов
fig, ax = plt.subplots(4, 6)
for i, axi in enumerate(ax.flat):
    axi.imshow(Xtest[i].reshape(62, 47), cmap='bone')
    axi.set(xticks=[], yticks=[])
    axi.set_ylabel(faces.target_names[yfit[i]].split()[-1],
                   color='black' if yfit[i] == ytest[i] else 'red')
fig.suptitle('Предсказанные имена; Неправильные ярлыки красным', size=14)
plt.show()

# Статистика по результатам классификации
from sklearn.metrics import classification_report
print('# Статистика по результатам классификации')
print(classification_report(ytest,yfit, target_names = faces.target_names))





