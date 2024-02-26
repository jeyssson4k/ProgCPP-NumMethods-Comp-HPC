import matplotlib.pyplot as plt

# Leer los datos de los archivos .txt
def leer_datos(nombre_archivo):
    with open(nombre_archivo, 'r') as file:
        lines = file.readlines()
    return [float(line.split()[0]) for line in lines], [float(line.split()[1]) for line in lines]

# Guardar los datos en arrays
x1, y1 = leer_datos('datos1.txt')
x2, y2 = leer_datos('datos2.txt')
x3, y3 = leer_datos('datos5.txt')

# Crear el gráfico con tres curvas
plt.plot(x1, y1, marker='*', label='SEED: 1', color="#000")
plt.plot(x2, y2, marker='*', label='SEED: 2', color="#F0202B")
plt.plot(x3, y3, marker='*', label='SEED: 5', color="#2012F8")

# Añadir etiquetas y título
plt.xlabel('Centro del Bin')
plt.ylabel('Valor en la PDF')
plt.title('PDF por semilla usada')

# Mostrar leyenda y gráfico
plt.legend()

# Guardar el gráfico como un archivo PDF
plt.savefig('normal_dist.pdf', format='pdf')

# Mostrar el gráfico
plt.show()