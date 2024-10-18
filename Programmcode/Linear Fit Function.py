import locale

import pandas as pd
import matplotlib.pyplot as plt
import re
from scipy.optimize import curve_fit
import numpy as np
import locale

from FitFunction import initial_guess

locale.setlocale(locale.LC_NUMERIC, 'de_DE.UTF-8')

#Variablendeklaration
thetadeg = np.array([39.7331, 46.130, 67.532, 81.295, 85.787, 103.64, 118.035, 123.06])

thetarad = np.radians(thetadeg / 2)

h = np.array([1, 2, 2, 3, 2, 4, 3, 4])
k = np.array([1, 0, 2, 1, 2, 0, 3, 2])
l = np.array([1, 0, 0, 1, 2, 0, 1, 0])
values = []

i  = 0

wavelength = 0.00000000015

x_axis = np.cos(thetarad)**2 / np.sin(thetarad)

#Brechnung von a mit zugehörigem hkl-Wert
for i in range(len(h)):
    hkl_value = h[i]**2+k[i]**2+l[i]**2
    append_value = ((wavelength * np.sqrt(h[i]**2 + k[i]**2 + l[i]**2))/(np.sin(thetarad[i])*2))
    values.append(append_value)

#In Numpy Array konvertieren
values = np.array(values)

#Definiere fit-fkt.
def linfit_function(x, A, B):
    return A*x+B

#führe fit durch
popt, pcov = curve_fit(linfit_function, x_axis, values)
A_opt, B_opt = popt
A_err, B_err = np.sqrt(np.diag(pcov))

x_fit = np.linspace(0, max(x_axis), 1000)
y_fit = linfit_function(x_fit, A_opt, B_opt)

#plotte fit
plt.plot(x_axis, values, 'o', label='Messdaten', color='red')
plt.plot(x_fit, y_fit, label=f'Linearer Fit: A={A_opt:.5e}, B={B_opt:.5e}', color='blue')

plt.xlabel(r'$\dfrac{\cos^2(\theta) }{ \sin(\theta)}$')
plt.ylabel(r'A in [nm]')
plt.title('Linearer Fit der Messdaten')

#Legende hinzufügen
plt.legend()

#Plot anzeigen
plt.grid(False)
plt.savefig('/Users/niccollingro/Desktop/FoPra lokal/Röntgenbeugung/Versuchsauswertung/Plots/Linear_a_fit.pdf', format='pdf')
plt.show()

#Fit-Parameter ausgeben
print(f"Gefittete Parameter:")
print(f"A = {A_opt:.5e} ± {A_err:.5e}")
print(f"B = {B_opt:.5e} ± {B_err:.5e}")




