import matplotlib.pyplot as plt

# Leer los datos del archivo .txt
with open('datos.txt', 'r') as file:
    lines = file.readlines()

# Extraer las columnas x e y
th = [float(line.split()[0]) for line in lines]
r = [float(line.split()[1]) for line in lines]
err = [float(line.split()[2]) for line in lines]
tf = [float(line.split()[3]) for line in lines]

t0 = tf[0]
s = [t0/t for t in tf]
print(f"t0: {t0}")
for i in range(16):
    print(f"th: {th[i]}, time: {tf[i]}, s: {s[i]}")
# Crear el gráfico con tres curvas
plt.plot(th, s, marker=',', label='Speedup', color="#000fff")


# Añadir etiquetas y título
plt.xlabel('Threads')
plt.ylabel('Speedup')
plt.title('Speedup executing Montecarlo in device 0')

# Mostrar leyenda y gráfico
plt.legend()

# Guardar el gráfico como un archivo PDF
plt.savefig('rk4.pdf', format='pdf')

# Mostrar el gráfico
plt.show()