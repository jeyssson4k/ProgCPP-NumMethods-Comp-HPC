import matplotlib.pyplot as plt
import numpy as np

def readfile(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    return [float(line.split()[0]) for line in lines], [float(line.split()[1]) for line in lines], [float(line.split()[2]) for line in lines], [float(line.split()[3]) for line in lines]
    

n, r, err, rt = readfile('rtx2050_cpu_serial.txt')
lb0 = 'f(x) = e^{-x^{2}}'
def n_r():
    plt.loglog(n, r, marker='*', label=f"${lb0}$", color="#FF6b00")
    plt.xlabel("n samples")
    plt.ylabel(f"$f(x)$")
    plt.title(f"Integral approximation of ${lb0}$ using Montecarlo Method")
    plt.legend(fontsize='small')
    plt.savefig('lr.pdf', format='pdf')
    plt.show()

def n_err():
    plt.loglog(n, err, marker='*', label=f"${lb0}$", color="#FF6b00")
    plt.xlabel("n samples")
    plt.ylabel(f"Associated error related to expected value")
    plt.title(f"Error computing integral of ${lb0}$ using Montecarlo Method")
    plt.legend(fontsize='small')
    plt.savefig('lr.pdf', format='pdf')
    plt.show()

def n_rt():
    plt.loglog(n, rt, marker='*', label=f"${lb0}$", color="#FF6b00")
    plt.xlabel("n samples")
    plt.ylabel(f"Time computing integral(s)")
    plt.title(f"Time computing integral of ${lb0}$ using Montecarlo Method")
    plt.legend(fontsize='small')
    plt.savefig('lr.pdf', format='pdf')
    plt.show()
