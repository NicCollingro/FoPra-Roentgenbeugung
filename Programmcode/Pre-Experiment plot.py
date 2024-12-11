import matplotlib.pyplot as plt
import numpy as np

def function(x, M):
    xcheck = np.sin(x / 2)
    result = np.where(xcheck == 0, 0, (np.sin(x * M / 2)**2) / (xcheck**2))
    return result
x_values = np.linspace(0.000001, 6.13, 100000)
x_values_max = x_values
x_values1 = np.linspace(6.5, 10, 10000)
for x in x_values1:
    x_values_max = np.append(x_values, x)


y_values_max = function(x_values_max, 2.3)
y_values1 = function(x_values , 2.3)
y_values2 = function(x_values1, 2.3)


plt.plot(x_values, y_values1)
plt.plot(x_values1, y_values2)
plt.title('')
plt.xlabel(r'h')
plt.ylabel(r'$\left| F \right|^2$')

plt.savefig('PFAD', format='pdf')

plt.show()