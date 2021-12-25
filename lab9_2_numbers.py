from sklearn.datasets import load_digits

import numpy as np


dd = load_digits()


# print(dd.DESCR)

m_d=dd['data']
tt=dd['target']
print(len(m_d))

import matplotlib.pyplot as plp


def show_cifr(index):
    plp.imshow(dd.images[index], cmap=plp.cm.gray_r, interpolation='nearest')
    plp.title('Цифра: ' + str(dd.target[index]))
    plp.show()

print(type(dd.images[0]))
show_cifr(4)

from sklearn import svm

df = np.array(
    [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 15.0, 15.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 15.0, 15.0, 15.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 15.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 15.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 15.0, 15.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 15.0, 15.0, 15.0, 15.0, 15.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ]
)

# print(type(df))
# print()
# print(m_d[1796])
m_d[1796] = df

# print(m_d[1796])



# print(df)
# print()
# print(dd.images[0])



svm = svm.SVC(gamma=0.001, C=100.)
svm.fit(m_d[:1790], tt[:1790])

print(type(df))

pr = svm.predict(m_d[1796:])
show_cifr(1795)

res = []



print(type(tt[1796:]))

pr = svm.predict(m_d[1795:])

res.append(pr[0])
print(res)