import matplotlib.pyplot as plt
import numpy as np

#leer el archivo
def readfile(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    return [float(line.split()[0]) for line in lines], [float(line.split()[1]) for line in lines], [float(line.split()[2]) for line in lines], [float(line.split()[3]) for line in lines], [float(line.split()[4]) for line in lines], [float(line.split()[5]) for line in lines]
    

color0 = "#ff6b00" #naranja
color1 = "#002950" #azul oscuro
t, x, y, k, pe, r = readfile('s1.txt')

#puedes cambiarlo por plt.loglog si necesitas escala loglog
plt.plot(t, k, marker='*', label="Pos X", color=color0) 
plt.plot(t, pe, marker='o', label="Pos Y", color=color1)
#este link sirve para ver los tipos de marker para el plot https://matplotlib.org/stable/api/markers_api.html
plt.xlabel("Tiempo (s)")
plt.ylabel("Energia (J)")
plt.title(f"Energia en el sistema")
plt.legend(fontsize='small')
plt.savefig('kpe_box2d.pdf', format='pdf')
plt.show()