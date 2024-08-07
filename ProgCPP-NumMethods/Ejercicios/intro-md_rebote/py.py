import numpy as np
import matplotlib.pyplot as plt

# Leer el archivo con los datos
# Asegúrate de usar la codificación correcta si no es UTF-8
h, f, c, rf, rc = np.genfromtxt('datos2.txt', unpack=True, encoding='ISO-8859-1')


plt.plot(h, c, '-o', label="distancia")
# Ajustar el diseño de la figura

# Mostrar la gráfica
plt.xlabel('h (tiempo)', fontsize=14)
plt.ylabel('Valores', fontsize=14)
plt.title('Gráfica de Datos', fontsize=16)
# Agregar leyenda
plt.legend()

# Escala logarítmica en
#plt.yscale('log')
#plt.xscale('log')

plt.savefig("Error_derivadas.pdf")