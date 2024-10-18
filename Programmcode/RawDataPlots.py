import matplotlib.pyplot as plt
import numpy as np
import locale

locale.setlocale(locale.LC_NUMERIC, 'de_DE.UTF-8')

#File Path
file_path='PFAD'

#array init
two_theta =[]
counts = []


#read in Data from file_path
with open(file_path, 'r') as file:
    for line in file:
        if line.strip() and 'Pos. [°2Th.]' not in line:
            parts = line.split()
            if len(parts) > 2:
                two_theta.append(float(parts[1]))
                counts.append(float(parts[3]))

#convert to numpy arrays
two_theta_np = np.array(two_theta)
counts_np = np.array(counts)


plt.plot(two_theta_np / 2, counts_np, label='CeO2-Messwerte')
plt.xlabel(r'$\theta$')
plt.ylabel(r'Counts')
plt.xlim(min(two_theta_np/2), max(two_theta_np/2))

plt.savefig('PFAD', format='PDF')
plt.show()