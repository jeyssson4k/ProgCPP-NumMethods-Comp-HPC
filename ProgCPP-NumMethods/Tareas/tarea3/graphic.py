import matplotlib.pyplot as plt
import numpy as np

# Leer los datos del archivo .txt
with open('data.txt', 'r') as file:
    lines = file.readlines()

h = [float(line.split()[0]) for line in lines]
err_forw = [float(line.split()[1]) for line in lines]
err_cent = [float(line.split()[2]) for line in lines]
plt.plot(h, err_forw, marker='*', label='Error porcentual [derivada forward]', color="#ab7f8b")
plt.plot(h, err_cent, marker='*', label='Error porcentual [derivada central]', color="#0bb90f")
plt.xscale("log")
plt.yscale("log")

# Añadir etiquetas y título
plt.xlabel('h')
plt.ylabel('Error porcentual')
plt.title('Error porcentual para las derivadas numéricas de la función 4xsin(x)')

# Mostrar leyenda y gráfico
plt.legend()

# Guardar el gráfico como un archivo PDF
plt.savefig('derivadas.pdf', format='pdf')

# Mostrar el gráfico
plt.show()
