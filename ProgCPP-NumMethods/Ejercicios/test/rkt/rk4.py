import matplotlib.pyplot as plt

# Leer los datos del archivo .txt
with open('r.txt', 'r') as file:
    lines = file.readlines()

# Extraer las columnas x e y
t = [float(line.split()[0]) for line in lines]
a1 = [float(line.split()[1]) for line in lines]
a2 = [float(line.split()[2]) for line in lines]
b1 = [float(line.split()[3]) for line in lines]
b2 = [float(line.split()[4]) for line in lines]


# Crear el gráfico con tres curvas
plt.plot(t, a1, marker='*', label='a1', color="#00f0ff")
plt.plot(t, a2, marker='*', label='a2', color="#fff000")
plt.plot(t, b1, marker='*', label='b1', color="#0f0f0b")
plt.plot(t, b2, marker='*', label='b2', color="#000fbf")


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