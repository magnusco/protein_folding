import numpy as np
import random as rdm
import matplotlib.pyplot as plt
import matplotlib as mpl

k_B = 1.38064852 * 10 ** (-23)
s = 0.005
d_max = 10**5


class Protein:

    # Define polymer length, temperature, gridsize, and initialize the polymer coordinates thereafter.
    def __init__(self, polymerLength, temperature, gridSize):
        self.N = polymerLength
        self.T = temperature
        self.N_t = gridSize
        # Set midpoint for even-length polymers.
        if polymerLength % 2 == 0:
            self.halfmono = int(polymerLength / 2)
            self.midpoint = int(gridSize / 2)
            self.startingpoint = int(self.midpoint - (polymerLength / 2))
            self.endpoint = round(self.startingpoint + polymerLength)
        # Set midpoint for odd-length polymers.
        else:
            self.halfmono = int(polymerLength / 2)
            self.midpoint = int(gridSize / 2)
            self.startingpoint = int(self.midpoint - (polymerLength / 2)) + 1
            self.endpoint = round(self.startingpoint + polymerLength)
        # Make initial polymer coordinates.
        self.coord = np.array([[self.midpoint, self.startingpoint]])
        for i in range(1, polymerLength):
            newcoord = np.array([[self.midpoint, self.startingpoint + i]])
            self.coord = np.append(self.coord, newcoord, axis=0)


class Grid:

    # Save the polymer in the system, draw empty grid of size specified by the protein object.
    def __init__(self, poly):
        self.poly = poly
        self.grid = np.zeros((self.poly.N_t, self.poly.N_t)).astype(np.int16)
        self.energy = 0
        self.drawPolyInGrid()
        self.U = self.get_U(poly.N_t)

    # Draw the current polymer in new grid, and save this as the system.
    def drawPolyInGrid(self):
        self.grid = np.zeros((self.poly.N_t, self.poly.N_t)).astype(np.int16)
        k = 1
        for monomer in self.poly.coord:
            self.grid[monomer[0]][monomer[1]] = 1
            k += 1
    def drawPolyInCopyGrid(self, poly):
        copygrid = np.zeros((self.poly.N_t, self.poly.N_t)).astype(np.int16)
        for monomer in poly:
            copygrid[monomer[0]][monomer[1]] = 1
        return copygrid

    # Print the grid as a matrix
    def printGrid(self):
        for i in range(0, self.poly.N_t):
            for j in range(0, self.poly.N_t):
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

        # Compute twist as polymer coordinates in newCoord
        if twistInfo[0] <= self.poly.halfmono:
            around = newCoord[twistInfo[0]-1]
            for i in range(twistInfo[0]-1, -1, -1):
                if twistInfo[1] == 1:
                    newCoord[i] = np.array([around[0] + (-around[1] + newCoord[i][1]),
                                            around[1] + (around[0] - newCoord[i][0])])
                else:
                    newCoord[i] = np.array([around[0] - (-around[1] + newCoord[i][1]),
                                            around[1] - (around[0] - newCoord[i][0])])
        else:
            around = newCoord[twistInfo[0]-1]
            for i in range(twistInfo[0]-1, self.poly.N, 1):
                if twistInfo[1] == 1:
                    newCoord[i] = np.array([around[0] + (around[1] - newCoord[i][1]),
                                            around[1] - (around[0] - newCoord[i][0])])
                else:
                    newCoord[i] = np.array([around[0] - (around[1] - newCoord[i][1]),
                                            around[1] + (around[0] - newCoord[i][0])])
        # Check if twist is legal -> draw in new copy of grid
        if self.legalTwist(newCoord):
            copygrid = self.drawPolyInCopyGrid(newCoord)
        else:
            return twisted

        # Check the new system energy -> save the copy grid as self.grid.
        E_original = self.calculateEnergy(self.poly.N, self.poly.coord, self.poly.N_t)
        E_twisted = self.calculateEnergy(self.poly.N, newCoord, self.poly.N_t)
        if E_twisted <= E_original:
            self.grid = copygrid
            self.poly.coord = newCoord
            twisted = 1
            self.energy = E_twisted
            return twisted
        elif self.monte_carlo(self.poly.T, E_original, E_twisted):
            self.grid = copygrid
            self.poly.coord = newCoord
            self.energy = E_twisted
            twisted = 1
            return twisted
        else:
            return twisted

    def monte_carlo(self, T, E_original, E_twisted):
        return np.random.random_sample() < np.exp(-(E_twisted - E_original) / (k_B * T))


    def nearestNeighbours(self, n, U, x, y, i, polymer_coords):
        # Calculates the energy contribution from the nearest
        # neighbour interactions on site (x, y).
        E = 0
        U_xy = U[x][y]
        if self.n_c(i, x + 1, y, polymer_coords):
            if self.isIn([x+1,y], polymer_coords):
                E = E - abs(U_xy - U[x + 1][y])
        if self.n_c(i, x - 1, y, polymer_coords):
            if self.isIn([x-1,y],polymer_coords):
                E = E - abs(U_xy - U[x - 1][y])
        if self.n_c(i, x, y + 1, polymer_coords):
            if self.isIn([x,y+1],polymer_coords):
                E = E - abs(U_xy - U[x][y + 1])
        if self.n_c(i, x, y - 1, polymer_coords):
            if self.isIn([x,y-1], polymer_coords):
                E = E - abs(U_xy - U[x][y + 1])
        return E

    def nearestNeighbours0(self, n, U, x, y, polymer_coords):
        E = 0
        U_xy = U[x][y]
        if self.n0(x + 1, y, polymer_coords):
            if self.isIn([x + 1, y], polymer_coords):
                E = E - abs(U_xy - U[x + 1][y])
        if self.n0(x - 1, y, polymer_coords):
            if self.isIn([x - 1, y], polymer_coords):
                E = E - abs(U_xy - U[x - 1][y])
        if self.n0(x, y + 1, polymer_coords):
            if self.isIn([x, y + 1], polymer_coords):
                E = E - abs(U_xy - U[x][y + 1])
        if self.n0(x, y - 1, polymer_coords):
            if self.isIn([x, y - 1], polymer_coords):
                E = E - abs(U_xy - U[x][y + 1])
        return E

    def nearestNeighboursL(self, n, U, x, y, N, polymer_coords):
        E = 0
        U_xy = U[x][y]
        if self.nL(N, x + 1, y, polymer_coords):
            if self.isIn([x + 1, y], polymer_coords):
                E = E - abs(U_xy - U[x + 1][y])
        if self.nL(N, x - 1, y, polymer_coords):
            if self.isIn([x - 1, y], polymer_coords):
                E = E - abs(U_xy - U[x - 1][y])
        if self.nL(N, x, y + 1, polymer_coords):
            if self.isIn([x, y + 1], polymer_coords):
                E = E - abs(U_xy - U[x][y + 1])
        if self.nL(N, x, y - 1, polymer_coords):
            if self.isIn([x, y - 1], polymer_coords):
                E = E - abs(U_xy - U[x][y + 1])
        return E

    def calculateEnergy(self, N, polymer_coords, N_t):
        # Returns the total energy of the grid.
        E = 0
        for i in range(1, N-2):
            E = E + 0.5 * self.nearestNeighbours(N_t, self.U, polymer_coords[i][0], polymer_coords[i][1],
                                                 i, polymer_coords)
            # multiply by 0.5 to avoid double counting.
        E += 0.5 * (self.nearestNeighboursL(N_t, self.U, polymer_coords[N-1][0], polymer_coords[N-1][1],
                                            N, polymer_coords)
                    + self.nearestNeighbours0(N_t, self.U, polymer_coords[0][0], polymer_coords[0][1], polymer_coords))
        return E

    def random_U(self):
        rvalue = (10.4 - 3.47)*10**(-21) * np.random.random_sample() - 10.4*10**(-21)
        return rvalue

    def n_c(self, i, x, y, coord):
        tf = ((np.array([x, y]) != coord[i+1]).all() and (np.array([x, y]) != coord[i-1]).all())
        return tf

    def n0(self, x, y, coord):
        return (coord[1] != [x, y]).all()

    def nL(self, i, x, y, coord):
        return (coord[i-1] != [x, y]).all()

    def get_U(self, N_t):
        U = np.zeros((N_t, N_t))
        for i in range(N_t):
            for j in range(N_t):
                U[i][j] = self.random_U()
        return U

    def printGraph(self,fig,i):
        ax = fig.add_subplot(1,3,i+1,facecolor='red')
        im = ax.imshow(self.grid,cmap='inferno')
        ax.set_title(str(i) + ' twists')
        ax.set_xlim([-0.5,14.5])
        ax.set_ylim([-0.5,14.5])
        return im

    def isIn(self, element, array):
        isin = False
        for elements in array:
            if (elements == element).all():
                isin = True
                break
        return isin

    def d(T):
        return (d_max * np.exp(-s * T)).astype(np.int32)



def plotMeanEnergy():
    temp = np.linspace(0, 1500, 100)
    meanenergy = []

    for t in temp:
        total
        polynom = Protein(15, t, 17)
        system = Grid(polynom)
        for i in range(0, system.d(t)):
            system.twist()
            if i > 15000:
                lastenergies.append(system.energy)
        meanenergy.append(sum(lastenergies)/float(len(lastenergies)))
        print(t)

    plt.plot(temp, meanenergy, 'k')
    plt.xlabel("Tenperature (Kelvin)")
    plt.ylabel("Mean protein energy")
    plt.show()


def teststuff():
    polynom = Protein(15, 1000, 16)
    system = Grid(polynom)
    for i in range(0, 100):
        system.twist()
    system.printGrid()

def figurestuff():
    polynom = Protein(15, 100000, 17)
    system = Grid(polynom)
    # system.printGrid()
    # fig = plt.figure()
    # fig.add_subplot(3,1,1)
    fig = plt.figure(facecolor='white')
    system.printGraph(fig, 0)
    plt.title('0 twists')
    i = 0
    while (i < 2):
        tf = system.twist()
        if (tf):
            i += 1
            im = system.printGraph(fig, i)
    # system.printGrid()
    cax, kw = mpl.colorbar.make_axes([ax for ax in fig.axes], fraction=.02)
    fig.colorbar(im, cax=cax, **kw)
    plt.show()


def main():
    plotMeanEnergy()
    return 0


main()

