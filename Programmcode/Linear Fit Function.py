
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import re
from scipy.optimize import curve_fit
import numpy as np
import locale

locale.setlocale(locale.LC_NUMERIC, 'de_DE.UTF-8')

pos_2Th_array = []
errors_array = []
#Dateipfad deklarieren
file_path='PFAD'

# Datei einlesen
with open(file_path, 'r') as file:
    # Zeilen der Datei lesen
    lines = file.readlines()

    # Schleife durch jede Zeile (ab der 2. Zeile, da die erste die Überschrift ist)
    for line in lines[1:]:
        # Extrahiere den Wert mit Klammern (Pos. [°2Th.] Spalte)
        match = re.search(r'(\d+\.\d+)\((\d+)\)', line)
        if match:
            value = float(match.group(1))  # Extrahiere den Hauptwert (ohne den Fehler)
            error = int(match.group(2)) * (10 ** (-len(match.group(1).split('.')[1])))  # Berechne den Fehler
            pos_2Th_array.append(value)
            errors_array.append(error)



#Variablendeklaration
pos_2Th_array = np.array(pos_2Th_array)
errors_array = np.array(errors_array)
errors_array = np.round(errors_array, decimals=4)
print(errors_array)
print(pos_2Th_array)
errors_array_rad = np.radians(errors_array / 2)
thetarad = np.radians(pos_2Th_array / 2)


h = np.array([1, 2, 2, 3, 2, 4, 3, 4, 4, 3, 5, 4, 5])
k = np.array([1, 0, 2, 1, 2, 0, 3, 2, 2, 3, 1, 4, 3])
l = np.array([1, 0, 0, 1, 2, 0, 1, 0, 2, 3, 1, 0, 1])
values = []
errors = []

i  = 0

wavelength = 0.00000000015

x_axis = np.cos(thetarad)**2 / np.sin(thetarad)

#Brechnung von a mit zugehörigem hkl-Wert
for i in range(len(h)):
    hkl_value = h[i]**2+k[i]**2+l[i]**2
    append_value = ((wavelength * np.sqrt(hkl_value))/(np.sin(thetarad[i])*2))
    values.append(append_value)
    error_appendvalue = append_value*(errors_array_rad[i]/np.tan(thetarad[i]))
    errors.append(error_appendvalue)
    print(append_value)

#In Numpy Array konvertieren
values = np.array(values)
errors = np.array(errors)

#Definiere fit-fkt.
def linfit_function(x, A, B):
    return A*x+B

#führe fit durch
popt, pcov = curve_fit(linfit_function, x_axis, values, sigma=errors)
A_opt, B_opt = popt
A_err, B_err = np.sqrt(np.diag(pcov))

x_fit = np.linspace(0, max(x_axis), 1000)
y_fit = linfit_function(x_fit, A_opt, B_opt)

#plotte fit
plt.errorbar(x_axis, values, yerr=errors,fmt= 'o', label='Messdaten', color='red')
plt.plot(x_fit, y_fit, label=f'Linearer Fit:', color='#004877')

plt.fill_between(x_fit,linfit_function(x_fit, A_opt - A_err, B_opt - B_err),linfit_function(x_fit, A_opt + A_err,B_opt + B_err),color='#004877', alpha=0.2, label='Fehlerband')

#plt.ylim(0.00000000038, max(y_fit)+0.000000000002)
def scientific_notation(y, pos):
    return f'{y * 1e10:.4f}'  # Multipliziert den Wert mit 10^10 und zeigt ihn an

# Formatter auf die y-Achse anwenden
plt.gca().yaxis.set_major_formatter(FuncFormatter(scientific_notation))
plt.xlabel(r'$\dfrac{\cos^2(\theta) }{ \sin(\theta)}$')
plt.ylabel(r'A in [$\AA$]')
plt.title('')

fit_info = [
    f'A = ({A_opt:.5e} ± {A_err:.5e})'.replace('.', ',').replace('e-10', r' $\cdot 10^{-10}$').replace('e-15', r' $\cdot 10^{-13}$').replace('e-14', r' $\cdot 10^{-14}$').replace('e-13', r' $\cdot 10^{-13}$') + r'nm $\cdot$ rad',
    f'B = ({B_opt:.5e} ± {B_err:.5e})'.replace('.', ',').replace('e-10', r' $\cdot 10^{-10}$').replace('e-15', r' $\cdot 10^{-13}$').replace('e-14', r' $\cdot 10^{-14}$').replace('e-13', r' $\cdot 10^{-13}$') + r'nm'
]

fit_text= '\n'.join(fit_info)

proxy_handle = plt.Line2D([0], [0], color='white', label=fit_text)

# Hinzufügen des Proxy-Handles für die Fit-Parameter
handles, labels = plt.gca().get_legend_handles_labels()
handles.append(proxy_handle)
labels.append(fit_text)

# Legende mit allen Handles aktualisieren
plt.legend(handles=handles, labels=labels, loc='upper left', frameon=True)

plt.tight_layout()

#Plot anzeigen
plt.grid(False)
plt.savefig('PFAD', format='pdf')
plt.show()

#Fit-Parameter ausgeben
print(f"Gefittete Parameter:")
print(f"A = {A_opt:.5e} ± {A_err:.5e}")
print(f"B = {B_opt:.5e} ± {B_err:.5e}")




