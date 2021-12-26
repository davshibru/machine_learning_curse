import matplotlib.pyplot as plt
import numpy as np

array = np.array([[0, 49, 24, 11, 19, 0, 0, 0, 0,  35, 0, 0, 24],
        [0, 0, 0, 22, 0, 15, 9, 39, 0, 4, 0, 0, 12],
        [0, 0, 0, 0, 0, 0, 3,  35, 0, 0,  33, 33, 28],
        [0, 24, 0, 0, 21, 28, 0, 6, 0, 37, 0, 0, 6],
        [21, 9, 0, 18, 0, 0, 0, 0, 0, 0, 3, 36, 0],
        [0, 0, 0, 0, 0, 0, 0, 16, 48, 0, 0, 22, 16],
        [0, 0, 9, 22, 16, 6, 0, 0, 31, 15, 0, 0, 34],
        [44, 0, 0, 0, 3, 41, 0, 0, 0, 0, 0, 0, 6],
        [0, 0, 32, 0, 0, 33, 2, 44, 0, 0, 0, 0, 0],
        [16, 19, 0, 0, 4, 12, 0, 0, 0, 0, 0, 0, 0],
        [0, 43, 0, 22, 0, 0, 8, 2, 0, 49, 0, 24, 0],
        [0, 19, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0],
        [28, 49, 39, 13, 14, 0, 0, 0, 0, 5, 19, 0, 0]])

# Построение графика по сумму строк матрицы
print('Построение графика по сумму строк матрицы')
print(f'суммы элементов строк матрицы - {np.sum(array,axis=1).tolist()}')
plt.plot(np.sum(array,axis=1), 'o')
plt.show()

# Построение графика по сумме элементов столбцов матрицы
print('Построение графика по сумме элементов столбцов матрицы')
print(f'суммы элементов столбцов матрицы - {np.sum(array,axis=0).tolist()}')
plt.plot(np.sum(array,axis=0), 'o')
plt.show()

# Построение графика по среднему значению каждой строки
print('Построение графика по среднему значению каждой строки')
print(f'По среднему значению каждой строки - {np.average(array, axis=1).tolist()}')
plt.plot(np.average(array, axis=1), 'o')
plt.show()

# Построение графика по среднему значению каждого столбца
print('Построение графика посреднему значению каждого столбца')
print(f'По среднему значению каждого столбца - {np.average(array, axis=0).tolist()}')
plt.plot(np.average(array, axis=0), 'o')
plt.show()


# plt.plot(array, 'o', color='red')
# plt.show()
#
# print(array)

# X,Y = np.meshgrid(np.arange(array.shape[1]), np.arange(array.shape[0]))
# s = [20*4**n for n in range(len(X))]
# plt.scatter(X.flatten(), Y.flatten(), s=s, c=array.flatten())
#
# plt.show()

fig = plt.figure(1)
for row in range(len(np.sum(array,axis=1).tolist())):
    print('\n')
    for colums in range(len(np.sum(array,axis=0).tolist())):
        plt.scatter(row, colums, s=(array[row][colums] * 10), alpha=0.5)
        #print(array[row][colums])
plt.colorbar()
plt.show()

