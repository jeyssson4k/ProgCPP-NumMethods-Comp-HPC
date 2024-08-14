import matplotlib.pyplot as plt

# Leer los datos del archivo .txt
with open('data.txt', 'r') as file:
    lines = file.readlines()

# Extraer las columnas x e y
t = [float(line.split()[0]) for line in lines]
n = [float(line.split()[1]) for line in lines]
dn = [float(line.split()[2]) for line in lines]


# Crear el gráfico con tres curvas
plt.plot(t, n, marker='*', label='rk4', color="#000")


# Añadir etiquetas y título
plt.xlabel('t')
plt.ylabel('n')
plt.title('rk4')

# Mostrar leyenda y gráfico
plt.legend()

# Guardar el gráfico como un archivo PDF
plt.savefig('rk4.pdf', format='pdf')

# Mostrar el gráfico
plt.show()