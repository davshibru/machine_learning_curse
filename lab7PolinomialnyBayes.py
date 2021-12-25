#подключение набора текстов
from sklearn.datasets import fetch_20newsgroups

# подключение библиотек с векторизатором TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# подключение библиотеки для построения матрицы коэфициентов
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

data = fetch_20newsgroups()
print(data.target_names)

categories = ['rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt']

# построение тренировочноо и тестового набора данных
train = fetch_20newsgroups(subset = 'train', categories = categories)
test = fetch_20newsgroups(subset = 'test', categories = categories)

#построение модели
model =make_pipeline(TfidfVectorizer(), MultinomialNB())

#обучение модели
model.fit(train.data, train.target)

# применение модели
labels = model.predict(test.data)

#построение матрицы коэфициентов сранения результатов обучения модели
mat = confusion_matrix(test.target, labels)
sns.heatmap(mat.T, square = True, annot = True, fmt = 'd', cbar = False, xticklabels = train.target_names,
                    yticklabels = train.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label')

plt.show()

# функция определяет текст для передаваемой строки



def predict_category(s, train = train, model = model):
    pred = model.predict([s])
    return train.target_names[pred[0]]


question = ['basket',
            'ball',
            'cap',
            'weels',
            'ice',
            'winter',
            'computer',
            'security',
            'bitcoin']

for s in question:
    print(s)
    print(predict_category(s))

s = ''
while (s != 'quit'):
    s = input("\nВведите что-нибуть\n")

    print(predict_category(s))