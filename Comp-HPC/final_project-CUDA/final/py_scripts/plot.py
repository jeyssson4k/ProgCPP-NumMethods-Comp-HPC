import matplotlib.pyplot as plt

# Leer los datos del archivo .txt
with open('data0.txt', 'r') as file:
    lines = file.readlines()

# Extraer las columnas x e y
tf = [float(line.split()[0]) for line in lines]
th = [float(line.split()[1]) for line in lines]

t0 = tf[0]
s = [t0/t for t in tf]

with open('data1.txt', 'r') as file:
    lines = file.readlines()

tf0 = [float(line.split()[0]) for line in lines]
th0 = [float(line.split()[1]) for line in lines]

t1 = tf0[0]
s0 = [t1/t for t in tf0]

with open('data2.txt', 'r') as file:
    lines = file.readlines()

tf1= [float(line.split()[0]) for line in lines]
th1 = [float(line.split()[1]) for line in lines]

t2 = tf1[0]
s1 = [t2/t for t in tf1]
# Crear el gráfico con tres curvas
plt.plot(th, s, marker=',', label='Speedup 0', color="#000fff")
plt.plot(th0, s0, marker=',', label='Speedup 1', color="#ff6b00")
plt.plot(th1, s1, marker=',', label='Speedup 2', color="#fb0bff")


# Añadir etiquetas y título
plt.xlabel('Threads')
plt.ylabel('Speedup')
plt.title('Montecarlo on device 0 using 506250000 samples')

# Mostrar leyenda y gráfico
plt.legend()

# Guardar el gráfico como un archivo PDF
plt.savefig('rk4.pdf', format='pdf')

# Mostrar el gráfico
plt.show()