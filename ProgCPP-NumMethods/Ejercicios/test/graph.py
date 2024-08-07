import matplotlib.pyplot as plt

# Leer los datos del archivo .txt
with open('data.txt', 'r') as file:
    lines = file.readlines()

# Extraer las columnas x e y
x = [float(line.split()[0]) for line in lines]
y = [float(line.split()[1]) for line in lines]


# Crear el gráfico con tres curvas
plt.plot(x, y, marker='*', label='Norma', color="#000")


# Añadir etiquetas y título
plt.xlabel('N')
plt.ylabel('Norma')
plt.title('Norma de los primos gemelos para cada N')

# Mostrar leyenda y gráfico
plt.legend()

# Guardar el gráfico como un archivo PDF
plt.savefig('norma.pdf', format='pdf')

# Mostrar el gráfico
plt.show()