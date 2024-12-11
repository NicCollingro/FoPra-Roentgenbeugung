import numpy as np
import re
import matplotlib.pyplot as plt
from scipy.stats import linregress  # Wichtig: Korrektes Importieren von linregress
from matplotlib.ticker import FuncFormatter

def cagliotiFunc (thetavalue):
    A = 0.00749
    B = -0.00924
    C = 0.01071
    return np.sqrt(A + B*np.tan(thetavalue) + C*(np.tan(thetavalue)**2))

# Datei einlesen
file_path = "/Users/niccollingro/Desktop/FoPra/Röntgenbeugung/GitRepo/Messdaten/PdAu-Reflexliste.txt"
data = []

wavelength = 1.54056e-10  # Korrekte Wellenlänge in Metern (1.54056 Å)

# Datei zeilenweise lesen und relevante Daten extrahieren
with open(file_path, "r") as file:
    for line in file:
        if line.strip() and not line.startswith("Nr.") and not line.startswith("Pos."):  # Überspringe Headerzeilen
            parts = line.split("\t")
            if len(parts) > 1:
                # Extrahiere den Wert für "Pos. [°2Th.]" (zweite Spalte)
                theta_value = parts[1].split("(")[0].strip()  # Entferne Unsicherheitsangaben in Klammern
                try:
                    data.append(float(theta_value))
                except ValueError:
                    print(f"Übersprungene Zeile: {line.strip()}")  # Debug-Ausgabe für fehlerhafte Zeilen

fwhm_values = []
fwhm_errors = []

# Datei zeilenweise lesen
with open(file_path, "r") as file:
    for line in file:
        # Überspringe Headerzeilen
        if line.strip() and not line.startswith("Nr.") and not line.startswith("Pos."):
            parts = line.split("\t")
            if len(parts) > 5:  # Stelle sicher, dass die rechte Halbwertsbreite vorhanden ist
                fwhm_right = parts[4].strip()  # Rechte Halbwertsbreite [°2Th.]

                # Extrahiere Wert und Fehler aus Klammern
                try:
                    # Extrahiere den numerischen Wert vor den Klammern
                    value = float(fwhm_right.split("(")[0].strip())
                    fwhm_values.append(value)

                    # Extrahiere den Fehler in den Klammern
                    match = re.search(r"\((\d+)\)", fwhm_right)
                    if match:
                        error = int(match.group(1)) * 10 ** -4  # Umrechnung von Hundertstelgrad
                        fwhm_errors.append(error)
                    else:
                        fwhm_errors.append(0)  # Kein Fehler angegeben
                except ValueError:
                    print(f"Ungültige Zeile übersprungen: {line.strip()}")

# Werte als numpy-Arrays speichern
fwhm_values = np.array(fwhm_values)
fwhm_errors = np.array(fwhm_errors)
theta_array = np.array(data)


# Umwandlung von Grad in Radianten
theta_array = np.radians(theta_array / 2)

# Berechnung von s0
s0 = (2 * np.sin(theta_array) / wavelength)**2

#create corrected width array
width_corrected = []
width_corrected_y = []

#correction width
for i in range(len(fwhm_values)):
    temp = (fwhm_values[i]**2) - (cagliotiFunc(theta_array[i])**2)
    width_corrected.append(temp)
    width_corrected_y.append(width_corrected[i] * (np.cos(theta_array[i])/wavelength)**2)

# Lineare Regression mit scipy.stats.linregress
slope, intercept, r_value, p_value, std_err = linregress(s0, width_corrected_y)

# Lineare Funktion zur Darstellung der Regression
def linear_func(x, slope, intercept):
    return slope * x + intercept

# Werte für die Regressionslinie berechnen
regression_line = linear_func(s0, slope, intercept)

# Ergebnisse der Regression ausgeben
print("Lineare Regression:")
print(f"Slope (Steigung): {slope:.14f}")
print(f"Intercept (Achsenabschnitt): {intercept:.6f}")

# Plot der Punkte und der Regressionslinie
def scientific_notation(y, pos):
    return f'{y*1e-20:.1f}'#f'{y * 1e10:.4f}'  # Multipliziert den Wert mit 10^10 und zeigt ihn an

def scientific_notationx(y, pos):
    return f'{y*1e-20:.1f}'#f'{y * 1e10:.4f}'  # Multipliziert den Wert mit 10^10 und zeigt ihn an

# Formatter auf die y-Achse anwenden
plt.gca().yaxis.set_major_formatter(FuncFormatter(scientific_notation))
plt.gca().xaxis.set_major_formatter(FuncFormatter(scientific_notationx))
plt.scatter(s0, width_corrected_y, color='#004877', label='Datenpunkte')
plt.plot(s0, regression_line, color='red', label='Regression', linestyle='-')
#plt.title(r'$\delta(s_0) vs. s_0$ mit Regressionslinie')
plt.title('')
plt.xlabel(r'$s_0 \, [\text{m}^{-20}]$')
plt.ylabel(r'$\delta(s_0) [\text{m}^{-20}]$')
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.savefig('/Users/niccollingro/Desktop/FoPra/Röntgenbeugung/Versuchsauswertung/Plots/PdAu_ds_gauss.pdf', format='PDF')
plt.show()