import matplotlib.pyplot as plt
import numpy as np

def readfile(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    return [float(line.split()[0]) for line in lines], [float(line.split()[1]) for line in lines], [float(line.split()[2]) for line in lines], [float(line.split()[3]) for line in lines]
    

dev0_color = "#c82e2e"
dev1_color = "#38510c"
dev2_color = "#76b900"
n, r, err, rt = readfile('data/local/cpu_serial.txt')
n, r1, err1, rt1 = readfile('data/sala2/cpu_serial.txt')
n2, r2, err2, rt2 = readfile('data/maxwell/cpu_serial.txt')
lb0 = 'f(x) = e^{-x^{2}}'
a = '\\left| 1 - \\frac{F(x)}{0.886227} \\right|'
def n_r():
    plt.loglog(n, r, marker='*', label=f"dev0", color=dev0_color)
    plt.loglog(n, r1, marker='o', label=f"dev1", color=dev1_color)
    plt.loglog(n, r2, marker='s', label=f"dev2", color=dev2_color)
    plt.xlabel("n samples")
    plt.ylabel(f"$F(x)$")
    plt.title(f"Integral approximation of ${lb0}$ using Montecarlo Method")
    plt.legend(fontsize='small')
    plt.savefig('n_vs_r.pdf', format='pdf')
    plt.show()

def n_err():
    plt.loglog(n, err, marker='*', label=f"dev0", color=dev0_color)
    plt.loglog(n, err1, marker='o', label=f"dev1", color=dev1_color)
    plt.loglog(n, err2, marker='s', label=f"dev2", color=dev2_color)
    plt.xlabel("n samples")
    plt.ylabel('$%s$'%a)
    plt.title(f"Error computing integral of ${lb0}$ using Montecarlo Method")
    plt.legend(fontsize='small')
    plt.savefig('n_vs_err.pdf', format='pdf')
    plt.show()

def n_rt():
    plt.loglog(n, rt, marker='*', label=f"dev0", color=dev0_color)
    plt.loglog(n, rt1, marker='o', label=f"dev1", color=dev1_color)
    plt.loglog(n, rt2, marker='s', label=f"dev2", color=dev2_color)
    plt.xlabel("n samples")
    plt.ylabel(f"Time computing integral (s)")
    plt.title(f"Time wasted computing integral of ${lb0}$ using Montecarlo Method")
    plt.legend(fontsize='small')
    plt.savefig('n_vs_rt.pdf', format='pdf')
    plt.show()

#n_r()
#n_err()
n_rt()