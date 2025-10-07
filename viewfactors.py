import numpy as np
import pandas as pd


def avionics_view_factors(Nrays, filepath):
    """
    Direct Python port of MATLAB avionicsViewFactors.m

    Assumptions:
    1. The fuselage is treated as a blackbox — rays that miss hit it.
    2. Avionics surfaces are rectangular.
    3. Rays are launched from random points with Lambertian directions.
    """

    # === Cylinder fuselage geometry (mm) ===
    R_fuse = 175.0
    L_fuse = 1154.0

    # === Load avionics table ===
    avionics = pd.read_excel(filepath, sheet_name="Avionics Geometry")
    Ncomp = avionics.shape[0]
    areas = avionics.iloc[:, 9].to_numpy(float)  # m²

    # === Initialize ===
    view_factor_matrix = np.zeros((Ncomp, Ncomp))
    total_sum = 0.0

    # === Loop through each component (except fuselage) ===
    for i in range(Ncomp - 1):

        # --- Determine which surfaces to skip ---
        if i <= 5:
            skip_me = list(range(0, 6))  # payload
        elif 6 <= i <= 11:
            skip_me = list(range(6, 12))  # power brick
        elif 12 <= i <= 17:
            skip_me = list(range(12, 18))  # auterion
        elif i == 18:
            skip_me = [18]  # GPS
        elif i == 19:
            skip_me = [19]  # ethernet
        elif 20 <= i <= 25:
            skip_me = list(range(20, 26))  # radio
        elif 26 <= i <= 31:
            skip_me = list(range(26, 32))  # battery1
        elif 32 <= i <= 37:
            skip_me = list(range(32, 38))  # battery2
        elif 38 <= i <= 43:
            skip_me = list(range(38, 44))  # battery3
        elif 44 <= i <= 49:
            skip_me = list(range(44, 50))  # battery4
        elif 50 <= i <= 55:
            skip_me = list(range(50, 56))  # silvus
        elif i == 56:
            skip_me = [56]  # tray
        else:
            skip_me = []

        # --- Initialize hit counter ---
        hits = np.zeros(Ncomp)

        # --- Geometry of current component ---
        xc, yc, zc, w, h, nx, ny, nz = avionics.iloc[i, 1:9]

        # --- Ray launching ---
        for _ in range(Nrays):

            # Random starting point on surface
            x, y, z = get_random_points(xc, yc, zc, w, h, nx, ny, nz)

            # Lambertian direction
            dx, dy, dz = lambertian_direction(nx, ny, nz)

            # First check intersection with fuselage
            hit_fuse, t_fuse = ray_cylinder_intersect(x, y, z, dx, dy, dz, R_fuse, L_fuse)

            nearest_t = np.inf
            hit_index = None

            if hit_fuse:
                nearest_t = t_fuse
                hit_index = Ncomp - 1  # fuselage index

            # --- Check intersections with other components ---
            for j in range(Ncomp):
                if j in skip_me:
                    continue

                xcj, ycj, zcj, wj, hj, nxj, nyj, nzj = avionics.iloc[j, 1:9]
                hit_rect, tj = ray_rectangle_intersect(
                    x, y, z, dx, dy, dz, xcj, ycj, zcj, wj, hj, nxj, nyj, nzj
                )
                if hit_rect and tj < nearest_t:
                    nearest_t = tj
                    hit_index = j

            # Record hit
            if hit_index is not None:
                hits[hit_index] += 1

        # --- Compute view factors for this component ---
        view_factor_matrix[i, :] = hits / Nrays

    # === Apply reciprocity for fuselage row ===
    fuselage_area = 2 * np.pi * R_fuse * L_fuse * 1e-6
    for i in range(Ncomp):
        if i == Ncomp - 1:
            view_factor_matrix[Ncomp - 1, i] = 1 - total_sum
        else:
            view_factor_matrix[Ncomp - 1, i] = (
                view_factor_matrix[i, Ncomp - 1] * (areas[i] / fuselage_area)
            )
        total_sum += view_factor_matrix[Ncomp - 1, i]

    print("View Factor Matrix:")
    print(view_factor_matrix)

    return view_factor_matrix


# ================================================================
# === Helper Functions (1:1 with MATLAB) =========================
# ================================================================

def lambertian_direction(nx, ny, nz):
    r1, r2 = np.random.rand(), np.random.rand()
    theta = 2 * np.pi * r1
    phi = np.arcsin(np.sqrt(r2))  # cosine-weighted

    if abs(nz) == 1:
        dx = np.cos(theta) * np.sin(phi)
        dy = np.sin(theta) * np.sin(phi)
        dz = np.cos(phi) * np.sign(nz)
    elif abs(ny) == 1:
        dx = np.cos(theta) * np.sin(phi)
        dy = np.cos(phi) * np.sign(ny)
        dz = np.sin(theta) * np.sin(phi)
    elif abs(nx) == 1:
        dx = np.cos(phi) * np.sign(nx)
        dy = np.sin(theta) * np.sin(phi)
        dz = np.cos(theta) * np.sin(phi)
    else:
        dx = dy = dz = 0.0
    return dx, dy, dz


def ray_cylinder_intersect(x0, y0, z0, dx, dy, dz, R, L):
    hit = False
    tmin = np.inf

    # Curved wall
    A = dy**2 + dz**2
    B = 2 * (y0 * dy + z0 * dz)
    C = y0**2 + z0**2 - R**2

    if A > 1e-12:
        roots = np.roots([A, B, C])
        roots = np.real(roots[np.isreal(roots)])
        roots = roots[roots > 0]
        for t in roots:
            xi = x0 + dx * t
            if abs(xi) <= L / 2 and t < tmin:
                hit = True
                tmin = t

    # End caps
    if abs(dx) > 1e-12:
        for cap in [-L / 2, L / 2]:
            tcap = (cap - x0) / dx
            if tcap > 0:
                ycap = y0 + dy * tcap
                zcap = z0 + dz * tcap
                if ycap**2 + zcap**2 <= R**2 and tcap < tmin:
                    hit = True
                    tmin = tcap
    return hit, tmin


def ray_rectangle_intersect(x0, y0, z0, dx, dy, dz,
                            xc, yc, zc, w, h, nx, ny, nz):
    hit = False
    t = np.inf
    if abs(nz) == 1:
        t = (zc - z0) / dz
        xi = x0 + dx * t
        yi = y0 + dy * t
        if abs(xi - xc) <= w / 2 and abs(yi - yc) <= h / 2:
            hit = True
    elif abs(ny) == 1:
        t = (yc - y0) / dy
        xi = x0 + dx * t
        zi = z0 + dz * t
        if abs(xi - xc) <= w / 2 and abs(zi - zc) <= h / 2:
            hit = True
    elif abs(nx) == 1:
        t = (xc - x0) / dx
        yi = y0 + dy * t
        zi = z0 + dz * t
        if abs(yi - yc) <= w / 2 and abs(zi - zc) <= h / 2:
            hit = True
    return hit, t


def get_random_points(xc, yc, zc, w, h, nx, ny, nz):
    if abs(nz) == 1:
        x = xc + (np.random.rand() - 0.5) * w
        y = yc + (np.random.rand() - 0.5) * h
        z = zc
    elif abs(ny) == 1:
        x = xc + (np.random.rand() - 0.5) * w
        y = yc
        z = zc + (np.random.rand() - 0.5) * h
    elif abs(nx) == 1:
        x = xc
        y = yc + (np.random.rand() - 0.5) * w
        z = zc + (np.random.rand() - 0.5) * h
    else:
        x = y = z = 0.0
    return x, y, z
