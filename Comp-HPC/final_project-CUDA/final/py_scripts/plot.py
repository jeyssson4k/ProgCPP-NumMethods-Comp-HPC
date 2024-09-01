import matplotlib.pyplot as plt

# Leer los datos del archivo .txt
with open('./intg-a4500/data0.txt', 'r') as file:
    lines = file.readlines()

# Extraer las columnas x e y
blxgr = [float(line.split()[0]) for line in lines]
thxbl = [float(line.split()[1]) for line in lines]
curand = [float(line.split()[2]) for line in lines]
map0 = [float(line.split()[3]) for line in lines]
host0 = [float(line.split()[4]) for line in lines]

tf = [x + y for x, y in zip(curand, map0)]
#t0 = 6228000
s = [tf[0]/t for t in tf]

# Leer los datos del archivo .txt
with open('./intg-a4500/data1.txt', 'r') as file:
    lines = file.readlines()

# Extraer las columnas x e y
blxgr1 = [float(line.split()[0]) for line in lines]
thxbl1 = [float(line.split()[1]) for line in lines]
curand1 = [float(line.split()[2]) for line in lines]
map1 = [float(line.split()[3]) for line in lines]
host1 = [float(line.split()[4]) for line in lines]

tf1 = [x + y for x, y in zip(curand1, map1)]
#t1 = 7325000
s1 = [tf1[0]/t for t in tf1]

# Leer los datos del archivo .txt
with open('./intg-a4500/data2.txt', 'r') as file:
    lines = file.readlines()

# Extraer las columnas x e y
blxgr2 = [float(line.split()[0]) for line in lines]
thxbl2 = [float(line.split()[1]) for line in lines]
curand2 = [float(line.split()[2]) for line in lines]
map2 = [float(line.split()[3]) for line in lines]
host2 = [float(line.split()[4]) for line in lines]

tf2 = [x + y for x, y in zip(curand2, map2)]
#t2 = 7011000
s2 = [tf2[0]/t for t in tf2]


e = []
e0 = []
e1 = []
for i in range(len(s)):
    e.append(s[i]/blxgr[i])
    e0.append(s1[i]/blxgr1[i])
    e1.append(s2[i]/blxgr2[i])


plt.plot(blxgr, e, marker=',', label='Efficiency 0', color="#76b900")
plt.plot(blxgr1, e0, marker=',', label='Efficiency 1', color="#ff6b00")
plt.plot(blxgr2, e1, marker=',', label='Efficiency 2', color="#000fff")
plt.axhline(y=0.75, color='#666666', linestyle='--')
# Añadir etiquetas y título
plt.xlabel('CUDA Blocks')
plt.ylabel('Efficiency')
plt.title('Montecarlo on NVIDIA RTX A4500 using 42250000 samples')

# Mostrar leyenda y gráfico
plt.legend()

# Guardar el gráfico como un archivo PDF
plt.savefig('e_vs_th_CUDA-6500.pdf', format='pdf')

# Mostrar el gráfico
plt.show()