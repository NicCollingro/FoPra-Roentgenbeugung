import matplotlib.pyplot as plt
import numpy as np

def function(x, M):
    xcheck = np.sin(x / 2)
    result = np.where(xcheck == 0, 0, (np.sin(x * M / 2)**2) / (xcheck**2))
    return result
x_values = np.linspace(0.000001, 6.13, 100000)
x_values1 = np.linspace(6.5, 10, 10000)
for x in x_values1:
    x_values = np.append(x_values, x)


y_values = function(x_values, 2.3)

plt.scatter(x_values, y_values, s=2)
plt.title('')
plt.xlabel(r'h')
plt.ylabel(r'$\left| F \right|^2$')

plt.savefig('PFAD', format='pdf')

plt.show()