import numpy as np

# n1 = input()
# n = int(n1)
# ar = []
#
# for i in range(n):
#     ar.append(i + 1)
#
# print(ar)
#
# revAr = list(reversed(ar))
#
# print(revAr)
#
# arrC = []
#
# for i in range(101):
#     arrC.append(i)
#
# print(arrC)
#
# arrF = []
# for i in arrC:
#     arrF.append(1.8 * i + 32)
#
# print(arrF)

# A = [12, 3, 34, 54, 1, 4, 43, 54, 66]
# x = 32
# n = len(A)
#
# sum = 0
#
# for i in range(n):
#     sum += A[i] * (x ** (i + 1))
#
# print(sum)

# def first(arr):
#     check = True
#     for i in range(len(arr) - 1):
#         if arr[i + 1] < arr[i]:
#             check = False
#             break
#     return check
# a = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# b = [4, 6, 3, 7, 3, 12, 45, 56, 6]
#
#
#
# if True == first(b):
#     print('последовательное')
# else:
#     print('не последовательное')
#
# n = 10
# arr = np.random.randint(100, size=n)
# print(arr)
# print(arr.mean())

# num = input('введите число, максимальное 100 - ')
# number = int(num)
# arr = np.random.randint(100, size=20)
# print(arr)
# print(len(arr[arr>number]))

# student_list = np.array([[10, 10, 10, 10, 10, 10], [10, 9, 8, 8, 10, 10], [6, 10, 4, 4, 3, 10], [2, 5, 2, 1, 10, 5], [6, 7, 8, 9, 7, 10]])
#
# print(student_list)
#
# print(np.mean(student_list, axis = 1, keepdims = True))

# student_list = np.array([[10, 10, 10, 10, 10, 10], [10, 9, 8, 8, 10, 10], [6, 10, 4, 4, 3, 10], [2, 5, 2, 1, 10, 5], [6, 7, 8, 9, 7, 10]])
#
# print(student_list)
# print(student_list.transpose())


# A = np.array([[10, 10, 10, 10, 10, 10], [10, 9, 8, 8, 10, 10], [6, 10, 4, 4, 3, 10], [2, 5, 2, 1, 10, 5], [6, 7, 8, 9, 7, 10]])
#
# print(A)
# print(np.mean(A, axis = 0))
#
# n = 3
# ar = []
# for i in range(n):
#     ar.append(i + 1)
#
# arr = np.array([ar])
#
# controler = n
# for i in range(n, n ** 2):
#     atractor = True
#
#
#     temp = []
#
#     if atractor:
#
#         for r in range(controler + n, controler, -1):
#             temp.append(r)
#             atractor = False
#     else:
#         for r in range(controler, controler + n):
#             temp.append(r)
#             atractor = True
#
#     arr = np.r_[arr, [temp]]
#     controler = controler + n
#     if controler >= n ** 2:
#         break
#
# print(arr)
#
# from scipy.linalg import pascal
#
# print(pascal(5, kind='lower'))

#from math import factorial
#
# # input n
# n = 5
# for i in range(n):
#     for j in range(n - i + 1):
#         # for left spacing
#         print(end=" ")
#
#     for j in range(i + 1):
#         # nCr = n!/((n-r)!*r!)
#         print(factorial(i) // (factorial(j) * factorial(i - j)), end=" ")
#
#     # for new line
#     print()