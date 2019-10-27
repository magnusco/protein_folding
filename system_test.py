from matplotlib import pyplot as plt
import numpy as np
import math as m

def mono15():
    temp = [100, 300, 500, 800, 1000, 1500]
    d = [75000, 72000, 46000, 32000, 20000, 15000]
    temp_reg = np.arange(0, 1500, 1)
    d_reg = []
    for t in temp_reg:
        d_reg.append(100000*m.exp(-0.0015*t))
    plt.plot(temp, d, 'k', label="required d")
    plt.plot(temp_reg, d_reg, 'r', label="d(T) (fitted curve)")
    plt.text(195, 90000, "10^5*exp(-0.0015*T)", fontsize=12, color='red')
    plt.grid('on')
    plt.xlabel("Temperature")
    plt.ylabel("d")
    plt.title("Number of needed calculations to achieve stability, 15 monomers")
    plt.legend()
    plt.show()

def mono30():
    temp = [100, 300, 500, 800, 1000, 1500]
    d = [40000, 36000, 30000, 26000, 19000, 17000]
    temp_reg = np.arange(0, 1500, 1)
    d_reg = []
    for t in temp_reg:
        d_reg.append(43000*m.exp(-0.00065*t))
    plt.plot(temp, d, 'k', label="required d")
    plt.plot(temp_reg, d_reg, 'r', label="d(T) (fitted curve)")
    plt.text(180, 40500, "43000*exp(-0.00065*T)", fontsize=12, color='red')
    plt.grid('on')
    plt.xlabel("Temperature")
    plt.ylabel("d")
    plt.title("Number of needed calculations to achieve stability, 30 monomers")
    plt.legend()
    plt.show()


def main():
    mono15()

main()