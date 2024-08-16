import matplotlib.pyplot as plt

# Leer los datos del archivo .txt
with open('datos.txt', 'r') as file:
    lines = file.readlines()

# Extraer las columnas x e y
t = [float(line.split()[0]) for line in lines]
r = [float(line.split()[1]) for line in lines]
# Crear el gráfico con tres curvas
plt.semilogy(t, r, marker=',', label='abs(r2-r1)', color="#3333ff")


# Añadir etiquetas y título
plt.xlabel('t')
plt.ylabel('abs(r2-r1)')
plt.title('abs(r2-r1) en un sistema de dos particulas variando Rz en 0.00001')

# Mostrar leyenda y gráfico
plt.legend()

# Guardar el gráfico como un archivo PDF
plt.savefig('abs_vs_t.pdf', format='pdf')

# Mostrar el gráfico
plt.show()
