import numpy as np
import random as rdm
import time
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
import matplotlib as mpl


k_B = 1.38064852 * 10 ** (-23)
d_max = 100000
s = 0.0015


class Protein:

    # Define polymer length, temperature, gridsize, and initialize the polymer coordinates thereafter.
    def __init__(self, polymerLength, gridSize):
        self.N = polymerLength
        self.gridSize = gridSize
        # Set midpoint for even-length polymers.
        if polymerLength % 2 == 0:
            self.halfmono = int(polymerLength / 2)
            self.midpoint = int(gridSize / 2)
            self.startingpoint = int(self.midpoint - (polymerLength / 2)) - 1
            self.endpoint = round(self.startingpoint + polymerLength)
        # Set midpoint for odd-length polymers.
        else:
            self.halfmono = int(polymerLength / 2)
            self.midpoint = int(gridSize / 2)
            self.startingpoint = round(self.midpoint - 1 - (polymerLength / 2)) + 1
            self.endpoint = round(self.startingpoint + polymerLength)
        # Make initial polymer coordinates.
        self.coord = np.array([[self.midpoint, self.startingpoint]])
        for i in range(1, polymerLength):
            newcoord = np.array([[self.midpoint, self.startingpoint + i]])
            self.coord = np.append(self.coord, newcoord, axis=0)
        self.totEnergy = 0
        self.U = self.get_U(self.N)

    def get_U(self, N_t):
        U = np.zeros((N_t, N_t))
        for i in range(N_t):
            for j in range(i + 2, N_t):
                U[i][j] += self.random_U()
        U = (U + np.transpose(U))
        return U

    def random_U(self):
        rvalue = (10.4 - 3.47) * 10 ** (-21) * np.random.random_sample() - 10.4 * 10 ** (-21)
        return rvalue


class Grid:

    # Save the polymer in the system, draw empty grid of size specified by the protein object.
    def __init__(self, poly, temperature):
        self.poly = poly
        self.T = temperature
        self.grid = np.zeros((self.poly.gridSize, self.poly.gridSize)).astype(np.int16)
        self.totEnergy = 0
        self.current_energy = 0
        self.legal_twist = 0

    # Draw the current polymer in new grid, and save this as the system.
    def drawPolyInGrid(self):
        self.grid = np.zeros((self.poly.gridSize, self.poly.gridSize)).astype(np.int16)
        k = 1
        for monomer in self.poly.coord:
            self.grid[monomer[0]][monomer[1]] = 1
            k += 1

    def drawPolyInCopyGrid(self, poly):
        copygrid = np.zeros((self.poly.gridSize, self.poly.gridSize)).astype(np.int16)
        k = 0
        for monomer in poly.coord:
            copygrid[monomer[0]][monomer[1]] = k+10
            k += 1
        return copygrid

    # Print the grid as a matrix
    def printGrid(self):
        self.drawPolyInGrid()
        for i in range(0, self.poly.gridSize):
            for j in range(0, self.poly.gridSize):
                print(self.grid[i][j], end=' ')
            print()
        print()

    # Compute a random monomer to twist around, and random direction (clockwise/anticlockwise).
    def randomTwist(self):
        monomer = rdm.randint(2, self.poly.N - 2)
        direction = rdm.randint(0, 1)
        return np.array([monomer, direction])

    # Check if the number of unique coordinates is equal to polymer length.
    def legalTwist(self, newCoord):
        return self.poly.coord.size == np.unique(newCoord, axis=0).size

    # Compute twist, if legal -> put into copy of grid.
    # Calculate energy, if energy is smaller -> save as new grid.
    def twist(self):
        twisted = 0
        newCoord = np.copy(self.poly.coord)
        twistInfo = self.randomTwist()
        E_original = self.calculateEnergy(self.poly.N, self.poly.coord)
        # Compute twist as polymer coordinates in newCoord
        if twistInfo[0] <= self.poly.halfmono:
            around = newCoord[twistInfo[0] - 1]
            for i in range(twistInfo[0] - 1, -1, -1):
                if twistInfo[1] == 1:
                    newCoord[i] = np.array([around[0] + (-around[1] + newCoord[i][1]),
                                            around[1] + (around[0] - newCoord[i][0])])
                else:
                    newCoord[i] = np.array([around[0] - (-around[1] + newCoord[i][1]),
                                            around[1] - (around[0] - newCoord[i][0])])
        else:
            around = newCoord[twistInfo[0] - 1]
            for i in range(twistInfo[0] - 1, self.poly.N, 1):
                if twistInfo[1] == 1:
                    newCoord[i] = np.array([around[0] + (around[1] - newCoord[i][1]),
                                            around[1] - (around[0] - newCoord[i][0])])
                else:
                    newCoord[i] = np.array([around[0] - (around[1] - newCoord[i][1]),
                                            around[1] + (around[0] - newCoord[i][0])])
        # Check if twist is legal -> draw in new copy of grid
        if not self.legalTwist(newCoord):
            self.legal_twist = 0
            return twisted

        self.legal_twist = 1
        # Check the new system energy -> save the copy grid as self.grid.
        E_twisted = self.calculateEnergy(self.poly.N, newCoord)
        if E_twisted <= E_original:
            self.poly.coord = newCoord
            self.totEnergy += E_twisted
            self.current_energy = E_twisted
            twisted = 1
            return twisted
        elif self.monte_carlo(self.T, E_original, E_twisted):
            self.poly.coord = newCoord
            self.totEnergy += E_twisted
            self.current_energy = E_twisted
            twisted = 1
            return twisted
        else:
            self.totEnergy += E_original
            return twisted

    def monte_carlo(self, T, E_original, E_twisted):
        return np.random.random_sample() < np.exp(-(E_twisted - E_original) / (k_B * T))

    def nearestNeighbours(self, N, U, x, y, j, polymer_coords):
        # Calculates the energy contribution from the nearest
        # neighbour interactions on site (x, y).
        E = 0
        for h in range(j + 1, N):
            if self.n_c(h, x + 1, y, polymer_coords):
                E += U[j][h]
                # print(h,j, U[j][h])
            elif self.n_c(h, x - 1, y, polymer_coords):
                E += U[j][h]
                # print(h,j, U[j][h])
            elif self.n_c(h, x, y + 1, polymer_coords):
                E += U[j][h]
                # print(h,j, U[j][h])
            elif self.n_c(h, x, y - 1, polymer_coords):
                E += U[j][h]
                # print(h,j, U[j][h])
        # print(E)
        return E

    def calculateEnergy(self, N, polymer_coords):
        # Returns the total energy of the grid.
        E = 0
        for i in range(0, N - 1):
            E = E + self.nearestNeighbours(N, self.poly.U, polymer_coords[i][0], polymer_coords[i][1],
                                           i, polymer_coords)
            # multiply by 0.5 to avoid double counting.
        return E

    def n_c(self, j, x, y, coord):
        tf = np.array_equal([x, y], coord[j])
        return tf

    def printGraph(self, fig, i, n, grSize):
        ax = fig.add_subplot(1, n, i + 1, facecolor='red')
        im = ax.imshow(self.grid, cmap='inferno')
        ax.set_title(str(i) + ' twists')
        ax.set_xlim([-0.5, grSize - 0.5])
        ax.set_ylim([-0.5, grSize - 0.5])
        return im

    def isIn(self, element, array):
        isin = False
        for elements in array:
            if (elements == element).all():
                isin = True
                break
        return isin

    def get_meanEnergy(self, d):
        return self.totEnergy / d


def printTime(start, end):
    diff = end - start
    h = int(diff // 3600)
    diff = diff - h * 3600
    m = int(diff // 60)
    s = diff - m * 60
    if h != 0:
        print("Runtime:", h, "hours,", m, 'min,', s, 'sec.')
    elif m != 0:
        print("Runtime:", m, 'min,', s, 'sec.')
    else:
        print("Runtime:", s, 'sec.')


def d(T):
    return (d_max * np.exp(-s * T)).astype(np.int32)


def plotMeanEnergy():
    N_t = 100
    temp = np.linspace(0.0001, 1500, N_t)
    d_vec = d(temp)
    mean_energy = np.zeros(len(temp))

    for t in range(0, N_t):
        polynom = Protein(15, 17)
        system = Grid(polynom, temp[t])
        legal_twist_count = 0
        for i in range(0, d_vec[t]):
            system.twist()
            legal_twist_count += system.legal_twist
        print(t)
        mean_energy[t] = (system.totEnergy / legal_twist_count)

    plt.plot(temp, mean_energy, 'k')
    plt.xlabel("Tenperature (Kelvin)")
    plt.ylabel("Mean energy (Joule)")
    plt.grid('on')
    plt.show()


def task2_2():
    poly = Protein(15, 17)
    system = Grid(poly, 0.0001)
    energy = [system.current_energy]
    twists = [0]
    legal_twist_count = 0
    while legal_twist_count < 5000:
        system.twist()
        if system.legal_twist:
            legal_twist_count += 1
            twists.append(legal_twist_count)
            energy.append(system.current_energy)
    plt.plot(twists, energy, 'k')
    plt.ylabel("Protein energy (Kelvin)")
    plt.xlabel("Number of twists")
    plt.grid('on')
    plt.show()

def task_1():
    N_t = 100
    T = np.linspace(10 ** -3, 1500, N_t)
    E = np.zeros(N_t)
    fig1 = plt.figure()
    polynom = Protein(15, 17)
    system = Grid(polynom, 0.0001)
    for j in range(0, 1000):
        system.twist()
        im = system.printGraph(fig1, 0, 2, 17)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # ax.set_ylim(-10**-16,10**-16)
    im = system.printGraph(fig1, 1, 2, 15)
    cax, kw = mpl.colorbar.make_axes([ax for ax in fig1.axes], fraction=.02)
    fig1.colorbar(im, cax=cax, **kw)
    plt.show()

def samplemat():
    polynom = Protein(15, 17)
    system = Grid(polynom, 0.0001)
    for i in range(0, 100):
        system.twist()
        plt.matshow(system.drawPolyInCopyGrid(system.poly))
        plt.show()
        plt.close()

def s_and_d():
    d = 100
    polynom = Protein(30, 32)
    system = Grid(polynom, 1500)
    mean_energy = []
    legal_twists = []
    twist_count = 0
    legal_twist_count = 0
    for i in range(1, d+1):
        system.twist()
        if system.legal_twist:
            legal_twist_count += 1
            mean_energy.append(system.totEnergy/legal_twist_count)
            legal_twists.append(legal_twist_count)
    plt.plot(legal_twists, mean_energy, 'k')
    plt.title("Mean energy as function of number of legal twists, T = 1500 K")
    plt.xlabel("Number of legal twists")
    plt.ylabel("Mean energy")
    plt.grid('on', 'both')
    plt.show()



def main():
    samplemat()
    return 0


main()