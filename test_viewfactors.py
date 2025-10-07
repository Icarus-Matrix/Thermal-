import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import viewfactors
from mpl_toolkits.mplot3d import Axes3D  # 3D plotting


def debug_visualization(filepath, Nrays):
    # Read avionics table
    avionics = pd.read_excel(filepath, sheet_name="Avionics Geometry").apply(pd.to_numeric, errors='ignore')


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(57):  # loop through components
        xc = avionics.iloc[i, 1]
        yc = avionics.iloc[i, 2]
        zc = avionics.iloc[i, 3]
        w  = avionics.iloc[i, 4]
        h  = avionics.iloc[i, 5]
        nx = avionics.iloc[i, 6]
        ny = avionics.iloc[i, 7]
        nz = avionics.iloc[i, 8]

        # Build surface grid depending on orientation
        if abs(nz) == 1:  # XY plane
            X, Y = np.meshgrid(np.linspace(xc - w/2, xc + w/2, 2),
                               np.linspace(yc - h/2, yc + h/2, 2))
            Z = np.full_like(X, zc)
        elif abs(ny) == 1:  # XZ plane
            X, Z = np.meshgrid(np.linspace(xc - w/2, xc + w/2, 2),
                               np.linspace(zc - h/2, zc + h/2, 2))
            Y = np.full_like(X, yc)
        elif abs(nx) == 1:  # YZ plane
            Y, Z = np.meshgrid(np.linspace(yc - w/2, yc + w/2, 2),
                               np.linspace(zc - h/2, zc + h/2, 2))
            X = np.full_like(Y, xc)
        else:
            continue

        # Plot surface
        ax.plot_surface(X, Y, Z, alpha=0.5)

        # Sample a few rays and plot them
        for n in range(Nrays):
            x, y, z = get_random_points(xc, yc, zc, w, h, nx, ny, nz)
            dx, dy, dz = lambertian_direction(nx, ny, nz)

            # Plot ray as arrow
            ax.quiver(x, y, z, dx, dy, dz, length=20, color='r', normalize=True)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()


def get_random_points(xc, yc, zc, w, h, nx, ny, nz):
    if abs(nz) == 1:  # z-normal surface
        x = xc + (np.random.rand() - 0.5) * w
        y = yc + (np.random.rand() - 0.5) * h
        z = zc
    elif abs(ny) == 1:  # y-normal surface
        x = xc + (np.random.rand() - 0.5) * w
        y = yc
        z = zc + (np.random.rand() - 0.5) * h
    elif abs(nx) == 1:  # x-normal surface
        x = xc
        y = yc + (np.random.rand() - 0.5) * w
        z = zc + (np.random.rand() - 0.5) * h
    return x, y, z


def lambertian_direction(nx, ny, nz):
    r1, r2 = np.random.rand(), np.random.rand()
    theta = 2 * np.pi * r1
    phi = np.arcsin(np.sqrt(r2))  # cosine-weighted distribution

    if abs(nz) == 1:  # z-normal
        dx = np.cos(theta) * np.sin(phi)
        dy = np.sin(theta) * np.sin(phi)
        dz = np.cos(phi) * nz
    elif abs(ny) == 1:  # y-normal
        dx = np.cos(theta) * np.sin(phi)
        dy = np.cos(phi) * ny
        dz = np.sin(theta) * np.sin(phi)
    elif abs(nx) == 1:  # x-normal
        dx = np.cos(phi) * nx
        dy = np.sin(theta) * np.sin(phi)
        dz = np.cos(theta) * np.sin(phi)
    return dx, dy, dz

if __name__ == "__main__":
    filepath = "C:/Users/BrianByrne/Desktop/Thermal Analysis/VIEWFACTOR GEOMETRY.xlsx"
    Nrays = 1000
    viewfactor_matrix=viewfactors.avionics_view_factors(Nrays,filepath)
    debug_visualization(filepath,Nrays)