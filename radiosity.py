#radiosity network solve 
#This script will solve the radiosity netowrk for the fuselage and the avionics

import numpy as np

def get_radiosity(number_surfs, emissivities, view_factors, temperatures, areas):

    #number of surfaces to loop through 
    N = number_surfs

    A = np.zeros((N, N))
    B = np.zeros(N)
    Rc = np.zeros(N)
    Rs = np.zeros((N, N))
    Eb = np.zeros(N)

    sigma = 5.67e-8  # Stefan–Boltzmann constant

    for i in range(N):
        Eb[i] = sigma * temperatures[i]**4  # blackbody emissive power

        if emissivities[i] == 1.0:  # blackbody shortcut
            A[i, :] = 0
            A[i, i] = 1
            B[i] = Eb[i]
            continue
        else:
            Rc[i] = (1 - emissivities[i]) / (emissivities[i] * areas[i]) #radiative resistance 
            A[i, i] += 1 / Rc[i] #contact resistances
            B[i] += Eb[i] / Rc[i] #black bodies B matrix 

        # Space resistances
        for j in range(N):
            if j == i:
                continue
            if view_factors[i, j] > 1e-12:
                Rs[i, j] = 1 / (areas[i] * view_factors[i, j]) #space resistances
                A[i, i] += 1 / Rs[i, j] #add space resistances to diagnol 
                A[i, j] -= 1 / Rs[i, j] #add space resistances to array 

    # Condition check
    if np.linalg.cond(A) > 1e12:  # reciprocal of rcond
        print("Warning: Matrix A is ill-conditioned. Check view factors or emissivities.")

    # Solve linear system
    J = np.linalg.solve(A, B)

    return J
    