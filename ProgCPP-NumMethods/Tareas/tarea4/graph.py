import matplotlib.pyplot as plt
import numpy as np

# Leer los datos del archivo .txt
with open('derivates.txt', 'r') as file:
    lines = file.readlines()

h     = [float(line.split()[0])  for line in lines]
c_o_1 = [float(line.split()[1])  for line in lines]
r_o_1 = [float(line.split()[2])  for line in lines]
c_o_2 = [float(line.split()[3])  for line in lines]
r_o_2 = [float(line.split()[4])  for line in lines]
c_o_3 = [float(line.split()[5])  for line in lines]
r_o_3 = [float(line.split()[6])  for line in lines]
c_o_4 = [float(line.split()[7])  for line in lines]
r_o_4 = [float(line.split()[8])  for line in lines]
c_o_5 = [float(line.split()[9])  for line in lines]
r_o_5 = [float(line.split()[10]) for line in lines]


plt.loglog(h, c_o_1, marker='*', label='Central O(1)')
plt.loglog(h, c_o_2, marker='*', label='Central O(2)')
plt.loglog(h, c_o_3, marker='*', label='Central O(3)')
plt.loglog(h, c_o_4, marker='*', label='Central O(4)')
plt.loglog(h, c_o_5, marker='*', label='Central O(5)')
plt.loglog(h, r_o_1, marker='*', label='Rich Central O(1)')
plt.loglog(h, r_o_2, marker='*', label='Rich Central O(2)')
plt.loglog(h, r_o_3, marker='*', label='Rich Central O(3)')
plt.loglog(h, r_o_4, marker='*', label='Rich Central O(4)')
plt.loglog(h, r_o_5, marker='*', label='Rich Central O(5)')


#plt.xscale("log")
#plt.yscale("log")

# Añadir etiquetas y título
plt.xlabel('h')
plt.ylabel('Error porcentual')
plt.title('Error porcentual para derivadas numéricas')

# Mostrar leyenda y gráfico
plt.legend(loc='lower left')

# Guardar el gráfico como un archivo PDF
plt.savefig('derivadas.pdf', format='pdf')

# Mostrar el gráfico
plt.show()
