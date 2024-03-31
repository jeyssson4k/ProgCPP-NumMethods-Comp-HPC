import json
import matplotlib.pyplot as plt
import numpy as np

def readfile(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    return [float(line.split()[0]) for line in lines], [float(line.split()[1]) for line in lines]

v, V_v = readfile('data.txt')
lr_v, lr_V_v = readfile("out.txt")
# Open the JSON file
with open('info.json') as f:
    # Load the JSON data
    data = json.load(f)

# Print the contents of the JSON file
print(data)

plt.plot(v, V_v, marker='*', label=data["cvlabel"], color=data["cvcolor"])
plt.plot(lr_v, lr_V_v, marker='*', label=data["lrlabel"], color=data["lrcolor"])
plt.xlabel(data["xlabel"])
plt.ylabel(data["ylabel"])
plt.title(data["title"])
plt.text(0.015, 0.775, f"h = {data['k']}\nphi = {data['intercept']:.6f} +- {data['dint']:.6f}", transform=plt.gca().transAxes)
plt.legend(fontsize='small')
plt.savefig('lr.pdf', format='pdf')
plt.show()
