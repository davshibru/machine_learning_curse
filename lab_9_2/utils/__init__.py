import pygame
from .settings import *
from .button import Button
pygame.init()
pygame.font.init()
from sklearn.datasets import load_digits

import numpy as np


dd = load_digits()



m_d=dd['data']
tt=dd['target']

import matplotlib.pyplot as plp


def show_cifr(index):
    plp.imshow(dd.images[index], cmap=plp.cm.gray_r, interpolation='nearest')
    plp.title('Цифра: ' + str(dd.target[index]))
    plp.show()


from sklearn import svm


svm = svm.SVC(gamma=0.001, C=100.)
svm.fit(m_d[:1790], tt[:1790])



def check(df):
    df = np.array(df)
    m_d[1796] = df
    pr = svm.predict(m_d[1796:])

    return str(pr[0])



