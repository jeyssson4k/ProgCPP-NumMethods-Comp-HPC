import matplotlib.pyplot as plt

# Leer los datos del archivo .txt
with open('datos.txt', 'r') as file:
    lines = file.readlines()

# Extraer las columnas x e y
t = [float(line.split()[0]) for line in lines]
dy = [float(line.split()[1]) for line in lines]
y = [float(line.split()[2]) for line in lines]


# Crear el gráfico con tres curvas
plt.plot(t, y, marker='o', label='y\'(t) en funcion de y(t)', color="#3333ff")


# Añadir etiquetas y título
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('y en funcion de t usando rk4')

# Mostrar leyenda y gráfico
plt.legend()

# Guardar el gráfico como un archivo PDF
plt.savefig('y_vs_t_rk4.pdf', format='pdf')

# Mostrar el gráfico
plt.show()