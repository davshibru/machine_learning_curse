# Шаг 11 импорт библеотек

import pygame
from .settings import *
from .button import Button
pygame.init()
pygame.font.init()
from sklearn.datasets import load_digits
import matplotlib.pyplot as plp

import numpy as np

# Шаг 12 подготовка датасета для тренировки

dd = load_digits()

# Шаг 13 разделение данных на “атрибуты” и “метки”

m_d=dd['data']
tt=dd['target']




# def show_cifr(index):
#     plp.imshow(dd.images[index], cmap=plp.cm.gray_r, interpolation='nearest')
#     plp.title('Цифра: ' + str(dd.target[index]))
#     plp.show()


# Шаг 14 создание и подготовка в модели (опорные вектора)
from sklearn import svm

svm = svm.SVC(gamma=0.001, C=100.)
svm.fit(m_d[:1790], tt[:1790])

# Шаг 15 создание функции котрая делает предсказание по масиву

def check(df):
    df = np.array(df)
    m_d[1796] = df
    pr = svm.predict(m_d[1796:])

    return str(pr[0])



