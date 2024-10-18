import locale

import pandas as pd
import matplotlib.pyplot as plt
import re
from scipy.optimize import curve_fit
import numpy as np
import locale

locale.setlocale(locale.LC_NUMERIC, 'de_DE.UTF-8')

# Funktion zum Extrahieren des Hauptwertes und des Fehlers aus den Klammern
def extract_value_and_error(value_str):
    match = re.match(r"([0-9.]+)\((\d+)\)", value_str)
    if match:
        value = float(match.group(1))
        error_digits = int(match.group(2))  # Fehler ist in der Klammer als ganze Zahl angegeben
        # Berechnen der Anzahl der Dezimalstellen im Hauptwert
        decimal_places = len(match.group(1).split('.')[-1])
        # Fehler auf die korrekte Größenordnung skalieren
        error = error_digits * (10 ** -decimal_places)
        return value, error
    else:
        return float(value_str), 0  # Kein Fehler angegeben


# Datei einlesen
file_path = 'PFAD'
data = pd.read_csv(file_path, delimiter='\t')

# Extrahieren der Werte und Fehler aus der Spalte "Pos. [°2Th.]" und "FWHM links [°2Th.]"
x_values = []
x_errors = []
y_values = []
y_errors = []

for index, row in data.iterrows():
    x_val, x_err = extract_value_and_error(str(row['Pos. [°2Th.]']))
    y_val, y_err = extract_value_and_error(str(row['FWHM links [°2Th.]']))

    x_values.append(x_val)
    x_errors.append(x_err)
    y_values.append(y_val)
    y_errors.append(y_err)

x_values = np.array(x_values)
y_values = np.array(y_values)
y_errors = np.array(y_errors)

theta = np.radians(x_values / 2)

def cogliotti_function(theta, A,B,C):
    return np.sqrt(A + B*np.tan(theta) + C*(np.tan(theta))**2)

initial_guess = [0, 0, 0]

popt, pcov = curve_fit(cogliotti_function, theta, y_values, sigma=y_errors, absolute_sigma=True, p0=initial_guess)

A_opt, B_opt, C_opt = popt

perr=np.sqrt(np.diag(pcov))

A_err, B_err, C_err = perr

plt.errorbar(x_values, y_values, yerr=y_errors, fmt='o', label='Messdaten mit Fehler', ecolor='red', capsize=5)

# Fitted curve plotten
x_fit = np.linspace(min(x_values), max(x_values), 500)
theta_fit = np.radians(x_fit / 2)
y_fit = cogliotti_function(theta_fit, A_opt, B_opt, C_opt)
plt.plot(x_fit, y_fit, label=f'Caglioti-Fit', color='blue')

textstr = '\n'.join((
    r'Ergebnisse des Fits für die Konstaten A,B und C',
    r'der Caglioti-Funktion : $\sqrt{A + B \cdot \tan{\theta} + C \cdot \left( \tan{\theta}\right)^2}$',
    r'$A = {:.5f} \pm {:.5f}$'.format(A_opt, A_err).replace('.',','),
    r'$B = {:.5f} \pm {:.5f}$'.format(B_opt, B_err).replace('.',','),
    r'$C = {:.5f} \pm {:.5f}$'.format(C_opt, C_err).replace('.',',')))

# Textfeld mit Konstanten und Fehlern hinzufügen
plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.2))


# Titel und Achsenbeschriftungen
plt.title('')
plt.xlabel(r'$\theta$ in [deg]')
plt.ylabel('Halbwertsbreite in [deg]')
plt.legend()
plt.grid(False)
plt.savefig('PFAD', format='pdf')
plt.show()

# Zeige die Parameter des Fits an
print(f"Gefittete Parameter:")
print(f"A = {A_opt:.5f} ± {A_err:.5f}")
print(f"B = {B_opt:.5f} ± {B_err:.5f}")
print(f"C = {C_opt:.5f} ± {C_err:.5f}")
