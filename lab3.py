import numpy as np
import pandas as pd

# FIO = pd.Series({0:'Шибру Давид', 1:'Лиров Зейнудин', 2:'Шакеев Бектур', 3:'Пенкин Сергей', 4:'Шибру Вераника', 5:'Гемоверова Гера'})
# exercise1 = pd.DataFrame({'fio': FIO})
# print(exercise1)
# SEX = pd.Series(['male', 'male', 'male', 'male', 'female', 'female'])
# GROUP = pd.Series(['IT-119', 'IT-119', 'BA-119', 'LAW-118', 'PED-119', 'IT-108'])
#
# exer2 = pd.DataFrame({'sex': SEX, 'group': GROUP})
# print(exer2)
#
# exercise2 = pd.DataFrame({'fio': FIO, 'sex': SEX, 'group': GROUP})
# print(exercise2)
#
# exercise2Final = pd.Series(GROUP, index = exercise2)
# exercise2Final = exercise2Final.reindex(exercise2)
# print(exercise2Final)
print()

# ex 1 -----------------------------------------------------------------------------------------------------
lastname = pd.Series(['Шибру', 'Лиров', 'Шакеев', 'Пенкин', 'Шибру', 'Лепров', 'Гемоверова'], name='lastname')
name = pd.Series(['Давид', 'Зейнудин', 'Бектур', 'Сергей', 'Вераника', 'Лик', 'Гера'], name='name')
FI = pd.DataFrame([lastname, name]).T
print(FI)

#ex 2 -----------------------------------------------------------------------------------------------------

SEX = pd.Series(['male', 'male', 'male', 'male', 'female', 'male', 'female'], name='sex')
GROUP = pd.Series(['IT-119', 'IT-119', 'BA-119', 'LAW-118', 'PED-119', 'BA-119', 'IT-108'], name='group')

Student = FI
Student['sex'] = SEX
Student['group'] = GROUP
print(Student.set_index(['lastname', Student.lastname]))

#ex3 -------------------------------------------------------------------------------------------------------------
print()

Student['gpa'] = pd.Series([2.4, 3.5, 3.8, 4, 4, 0.8, 2.7])

print(Student.sort_values(by='gpa'))

# #Получение максимального gpa(Баллы)
# otlichniks_gpa_max = Student.max().get("gpa")
# print(f'максимальный бал - {otlichniks_gpa_max}')
#
# #Получение минимального gpa(Баллы)
# otlichniks_gpa_mix = Student.min().get("gpa")
# print(f'минимальный бал - {otlichniks_gpa_max}')

gpa_dly_Neuspevauschih = 2
gpa_dly_Otlichnikov = 3.5


#массив в котором будут хранится айди отличников
indexOfOtlichnecs = []

#массив в котором будут хранится айди неуспевающих
indexOfNeuspevauschih = []

print()

#поиск айди отличников и неуспевающих

for i in range(len(Student.get('gpa'))):

    if Student.get("gpa")[i] >= gpa_dly_Otlichnikov:
        indexOfOtlichnecs.append(i)
    if Student.get("gpa")[i] <= gpa_dly_Neuspevauschih:
        indexOfNeuspevauschih.append(i)



print('список отличников:')
for i in indexOfOtlichnecs:
    print(f'{Student.get("lastname")[i]} {Student.get("name")[i]} - {Student.get("gpa")[i]}')

print()

print('список неуспевающих')
for i in indexOfNeuspevauschih:
    print(f'{Student.get("lastname")[i]} {Student.get("name")[i]} - {Student.get("gpa")[i]}')

#ex 5 --------------------

print()

groups = {}

for i in Student.get('group'):
    if i not in groups:
        groups[i] = []

for i in Student.values:
    groups[i[3]].append(i[4])


maxAverGpa = 0
maxAverGpaTitle = ''

for i in groups:
    if np.mean(groups[i]) > maxAverGpa:
        maxAverGpa = np.mean(groups[i])
        maxAverGpaTitle = i

        if maxAverGpa == int(np.mean(groups[i])) and maxAverGpaTitle.find(i) < 0:
            maxAverGpaTitle += f' {i}'


print(f'Группа с самым высоким средним балом {maxAverGpaTitle}')

