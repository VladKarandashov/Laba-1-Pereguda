import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import chi2
from prettytable import PrettyTable
from math import exp

# задаём начальные глобальные константы
mu = 9 # любое (0; 10]
lamda = 3 # любое (0; 10]
n = 100_0000 # чем больше тем лучше

print("Распределение с параметрами:")
print(f"mu = {mu}")
print(f"lamda = {lamda}")

#создаём начальные рандомные равномерно-распределённые числа
initial_uniform_numbers = list(np.random.uniform(size=n))

# обратная функция (получили на бумаге расчётами)
def inverse_function(y):
    x = mu - lamda*math.log((1/y) - 1)
    return x

# применяем обратную функцию и получаем правильно распределённые числа
distribution_row = list(map(inverse_function, initial_uniform_numbers))

# посчитаем моменты распределения
x_sred = (1/n)*sum(distribution_row)
corrected_dispersion_s2 = (1/(n-1)) * sum([(x - x_sred)**2 for x in distribution_row])
print("x-среднее", x_sred)
print("исправленная дисперсия S^2", corrected_dispersion_s2)
print("мат ожидание (должно совпасть с mu):")
print(x_sred-1.96*(corrected_dispersion_s2/n)**0.5, end=' < a < ')
print(x_sred+1.96*(corrected_dispersion_s2/n)**0.5)
print("Дисперсия:")
print((n-1)*corrected_dispersion_s2/chi2.ppf(0.5+0.95/2, n-1), end=' < σ^2 < ')
print((n-1)*corrected_dispersion_s2/chi2.ppf(0.5-0.95/2, n-1))

# сортируем правильно-распределённые числа по возрастанию
distribution_row.sort()

# эмпирическая функция распределения
# должна вернуть (кол-во элементов <= передаваемого)/n
# по факту, так как массив отсортирован - возвращает (i+1)/n
# применяем её к числам distribution_row
empirical_F_y = []
for i in range(len(distribution_row)):
    empirical_F_y.append((i+1)/n)

# строим график empirical_F_y(distribution_row)
fig, ax = plt.subplots()
ax.plot(distribution_row, empirical_F_y, label='line 1')
ax.set_title('Функция распределения')
ax.grid()
plt.show()

# РАССЧИТЫВАЕМ ГИСТОГРАММУ

#округление в сторону нечётного
def roundToNechet(number):
    if int(number) % 2 == 0:
        return int(number)+1
    return int(number)
#предварительное количество интервалов группировки на которое
#должна быть разбита область значений Χ
k = roundToNechet(1+3.2*math.log(n))
#Длина интервала
h = (max(distribution_row) - min(distribution_row))/k
# округляем длину интервала
if h<1:
    h = 1
else:
    (int(h*10)+1)/10
#найдём середину области изменения выборки (центр гистограммы)
C = (max(distribution_row) + min(distribution_row))/2
#найдём начало гистограммы
a = C - (k/2)*h
#просчитываем данные для графика-гистограммы
x_bar_chart = [(a+h*i, a+h*i+h) for i in range(0, k)]
x_mid_columns = [(p[0]+p[1])/2 for p in x_bar_chart]
countPointsInGap = lambda p: len([x for x in distribution_row if x>p[0] and x<=p[1]])
y_bar_chart = [countPointsInGap(p)/n for p in x_bar_chart]


#Построим отдельно гистограмму
fig, ax = plt.subplots()
ax.bar(x_mid_columns, y_bar_chart, width=h,
       color = 'chartreuse',
       edgecolor = 'darkblue',
       linewidth = 1)
ax.grid()
ax.set_title('Гистограмма для дальнейшей плотности вероятности')
plt.show()

#ПОСТРОИМ ЭМПИРИЧЕСКУЮ ПЛОТНОСТЬ ИСПОЛЬЗУЯ ГИСТОГРАММУ
fig, ax = plt.subplots()
ax.bar(x_mid_columns, y_bar_chart, width=h,
       color = 'chartreuse',
       edgecolor = 'darkblue',
       linewidth = 1)
ax.plot(x_mid_columns, y_bar_chart, 'r', label='line 1', linewidth=2, antialiased=True)
ax.grid()
ax.set_title('Плотность вероятности')
plt.show()

def myR(x):
    return int(x*10000)/10000

#для каждого интервала ищем вероятность попасть туда
x = PrettyTable()
Y = 0
P = 0
x.field_names = ["ai", "bi", "ni", "pi", "n*pi", "Отклонение"]

#посчитаем первый интервал отдельно
a = x_bar_chart[0][0] # в формуле оно -inf
b = x_bar_chart[0][1]
ni = countPointsInGap((a, b))
pi = 1/(1+exp((mu-b)/lamda))
npi = n*pi
y = ((ni - npi)**2)/npi
Y += y
P += pi
x.add_row(["-inf", myR(b), ni, myR(pi), myR(npi), myR(y)])


for i in range(1, len(x_bar_chart)-1):
    interval = x_bar_chart[i]
    a = interval[0]
    b = interval[1]
    ni = countPointsInGap(interval)
    pi = 1/(1+exp((mu-b)/lamda)) - 1/(1+exp((mu-a)/lamda))
    npi = n*pi
    y = ((ni - npi)**2)/npi
    Y += y
    P += pi
    x.add_row([myR(a), myR(b), ni, myR(pi), myR(npi), myR(y)])

#посчитаем последний интервал отдельно
a = x_bar_chart[-1][0] 
b = x_bar_chart[-1][1] # в формуле оно +inf
ni = countPointsInGap((a, b))
pi = 1 - 1/(1+exp((mu-a)/lamda))
npi = n*pi
y = ((ni - npi)**2)/npi
Y += y
P += pi
x.add_row([myR(a), "+inf", ni, myR(pi), myR(npi), myR(y)])

print(x)
print("Проверяем суммарную вероятность (нормировку):")
print(P)
print()
print("хи-квадрат вычисленное", Y)
print("критическая точка хи-квадрат", chi2.ppf(0.95, k-1))
print("гипотеза подтверждается")



input()
    
    























