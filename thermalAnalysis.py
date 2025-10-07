import numpy as np
import pandas as pd
import viewfactors
import radiosity
from CoolProp.CoolProp import PropsSI

sigma = 5.67e-8  # Stefan-Boltzmann

def thermal_analysis(altitude, latitude, day, solarTime, airSpeed, totalTime, vent_velocity,
                     filepath="C:/Users/BrianByrne/Desktop/Thermal Analysis/VIEWFACTOR GEOMETRY.xlsx"):
    """
    Python port of thermalAnalysis.m
    Returns a tuple in the same order as MATLAB outputs.
    """

    # ===== worksheet =====
    # Excel units: qgen [W/m^2 per surface], areas [m^2], k [W/m-K], thickness [m], rho [kg/m^3], Cp [J/kg-K]
    avionics = pd.read_excel(filepath, sheet_name="Avionics Geometry")

    # ===== constants =====
    P0 = 101325.0      # Pa at sea level
    Rair = 287.05      # J/kg-K

    # ===== fuselage geometry =====
    fuselageLength = 1154.0 * 1e-3   # m
    fuselageSurfaceArea = 2.486      # m^2
    fuselageMass = 1150.0 * 0.002    # kg

    # ===== heat sink parameters =====
    heatsinkthickness = 0.0015  # m
    heatsinkK = 240.0

    # ===== parameters from excel sheet =====
    Nsurfaces = avionics.shape[0]
    density = avionics.iloc[:, 10].to_numpy(dtype=float)          # col 11
    specificHeats = avionics.iloc[:, 11].to_numpy(dtype=float)    # col 12
    emissivities = avionics.iloc[:, 14].to_numpy(dtype=float)     # col 15
    absorptivities = avionics.iloc[:, 15].to_numpy(dtype=float)   # col 16
    thickness = avionics.iloc[:, 17].to_numpy(dtype=float)        # col 18
    k = avionics.iloc[:, 12].to_numpy(dtype=float)                # col 13
    areas = avionics.iloc[:, 9].to_numpy(dtype=float)             # col 10
    qgen = avionics.iloc[:, 16].to_numpy(dtype=float)             # col 17
    avionicTemperatures = avionics.iloc[:, 13].to_numpy(dtype=float)  # col 14 initial T

    # In MATLAB, Twalloutside = avionicTemperatures(Nsurfaces)
    Twalloutside = avionicTemperatures[Nsurfaces - 1]
    Twallinside = Twalloutside
    avionicTemperatures[Nsurfaces - 1] = Twallinside  # keep consistent

    # ===== time stepping parameters =====
    delta = 0.01            # s
    storeEverySec = 10      # s
    storeEverySteps = int(round(storeEverySec / delta))

    nsteps = int(np.floor(totalTime / delta)) + 1
    nstore = int(np.floor((nsteps - 1) / storeEverySteps)) + 1

    # ===== surface groups (convert MATLAB 1-based to Python 0-based) =====
    def rng(a, b):  # inclusive range in MATLAB becomes inclusive here (convert to 0-based)
        return list(range(a - 1, b))

    r2sPayload    = rng(1, 6)
    r2sPowerbrick = rng(7, 12)
    r2sAuterion   = rng(13, 18)
    r2sGPS        = [19 - 1]
    r2sethernet   = [20 - 1]
    r2sRadio      = rng(21, 26)
    r2sBattery1   = rng(27, 32)
    r2sBattery2   = rng(33, 38)
    r2sBattery3   = rng(39, 44)
    r2sBattery4   = rng(45, 50)
    r2sSilvus     = rng(51, 56)
    idx_fuse      = Nsurfaces - 1  # last index

    # ===== preallocation =====
    temperatureTrackArray = np.zeros((Nsurfaces + 1, nstore))  # +1 to store outside wall T
    timeArray = np.zeros(nstore)

    qnetArray = np.zeros((Nsurfaces, nstore))
    outCondArray = np.zeros(nstore)
    convectionArray = np.zeros(nstore)
    absSolarIrradianceArray = np.zeros(nstore)
    radiosityArray = np.zeros(nstore)
    insideConductionArray = np.zeros(nstore)
    TwalloutsideTemp = np.zeros(nstore)
    Qwallout = np.zeros(nstore)

    # per-component temps & net powers
    payloadTemp = np.zeros(nstore);        payloadNetPower = np.zeros(nstore)
    powerbrickTemp = np.zeros(nstore);     powerbrickNetPower = np.zeros(nstore)
    auterionTemp = np.zeros(nstore);       auterionNetPower = np.zeros(nstore)
    gpsTemp = np.zeros(nstore);            gpsNetPower = np.zeros(nstore)
    ethernetTemp = np.zeros(nstore);       ethernetNetPower = np.zeros(nstore)
    radioTemp = np.zeros(nstore);          radioNetPower = np.zeros(nstore)
    battery1Temp = np.zeros(nstore);       battery1NetPower = np.zeros(nstore)
    battery2Temp = np.zeros(nstore);       battery2NetPower = np.zeros(nstore)
    battery3Temp = np.zeros(nstore);       battery3NetPower = np.zeros(nstore)
    battery4Temp = np.zeros(nstore);       battery4NetPower = np.zeros(nstore)
    TwallinsideTemp = np.zeros(nstore);    TwallinsideNetPower = np.zeros(nstore)
    silvusTemp = np.zeros(nstore);         silvusNetPower = np.zeros(nstore)

    qnet = np.zeros(Nsurfaces)

    # derivatives
    dTpayload = np.zeros(nstore)
    dTpowerbrick = np.zeros(nstore)
    dTauterion = np.zeros(nstore)
    dTgps = np.zeros(nstore)
    dTethernet = np.zeros(nstore)
    dTradio = np.zeros(nstore)
    dTbattery1 = np.zeros(nstore)
    dTbattery2 = np.zeros(nstore)
    dTbattery3 = np.zeros(nstore)
    dTbattery4 = np.zeros(nstore)
    dTSilvus = np.zeros(nstore)
    dTwallinside = np.zeros(nstore)
    dTwalloutside = np.zeros(nstore)
    dTdt = np.zeros(Nsurfaces)

    # ===== view factors =====
    Nrays = 2000
    # expects avionics_view_factors to read the same Excel and include fuselage as last surface
    viewfactorMatrix = viewfactors.avionics_view_factors(Nrays, filepath)  # shape (Nsurfaces, Nsurfaces)

    # ===== air & convection =====
    airDensity, airTemperature, pressure = getAirProperties(altitude, Rair)
    htc = getHTC(pressure, airTemperature, airDensity, fuselageLength, airSpeed)
    internalHTC = getinternalHTC(vent_velocity, airDensity, airTemperature, pressure)

    # ===== time marching =====
    z = 0  # storage index (0-based)

    for w in range(nsteps):
        tcurrent = (w) * delta

        # update solar & radiosity inputs
        solarIrradiance, albedoLoad = getSolarLoad(day, pressure, P0, solarTime, latitude)
        J = getRadiosity(Nsurfaces, emissivities, viewfactorMatrix, avionicTemperatures)  # W/m^2

        # compute qnet from radiosity: q_i = sum_j F_ij * J_j - J_i
        # viewfactorMatrix is F_ij with row i summing to 1
        qnet[:] = viewfactorMatrix.dot(J) - J

        for i in range(Nsurfaces):
            if i == idx_fuse:
                # ===== outside wall energy balance =====
                outsideConduction = fuselageSurfaceArea * (k[i] / thickness[i]) * (Twallinside - Twalloutside)
                convection = fuselageSurfaceArea * htc * (Twalloutside - airTemperature)
                absSolarIrradiance = (fuselageSurfaceArea / 2.0) * (absorptivities[i] * (solarIrradiance + albedoLoad))
                emittedRadiation = (emissivities[i] * sigma * fuselageSurfaceArea * (Twalloutside**4 - airTemperature**4))
                Qout = absSolarIrradiance + outsideConduction - convection - emittedRadiation

                dTdt_out = Qout / (fuselageMass * specificHeats[i])
                Twalloutside = (delta / (fuselageMass * specificHeats[i])) * Qout + Twalloutside

                # ===== inside wall energy balance =====
                insideConduction = fuselageSurfaceArea * (k[i] / thickness[i]) * (Twallinside - Twalloutside)
                dTdt_in = ((qnet[i] * fuselageSurfaceArea) - insideConduction) / (fuselageMass * specificHeats[i])
                Twallinside = (delta / (fuselageMass * specificHeats[i])) * ((qnet[i] * fuselageSurfaceArea) - insideConduction) + Twallinside

                avionicTemperatures[idx_fuse] = Twallinside

                if (w % storeEverySteps) == 0:
                    temperatureTrackArray[i, z] = Twallinside
                    temperatureTrackArray[i + 1, z] = Twalloutside  # extra row for outside wall
                    qnetArray[i, z] = qnet[i]
                    outCondArray[z] = outsideConduction
                    convectionArray[z] = convection
                    absSolarIrradianceArray[z] = absSolarIrradiance
                    radiosityArray[z] = emittedRadiation
                    insideConductionArray[z] = insideConduction
                    timeArray[z] = (w * delta) / 3600.0
                    Qwallout[z] = Qout
                    dTwallinside[z] = dTdt_in
                    dTwalloutside[z] = dTdt_out

            else:
                # ===== avionic energy balance =====
                qavionicConvection = internalHTC * (avionicTemperatures[i] - airTemperature)  # W/m^2

                if i == (5) or i == (55):  # indices 5 and 55 in Python (payload face & silvus face per your rule)
                    qConduction = (heatsinkK / heatsinkthickness) * (avionicTemperatures[i] - airTemperature)
                    avionicTempNew = (delta / (density[i] * thickness[i] * specificHeats[i])) * \
                                     (qnet[i] + qgen[i] - qavionicConvection - qConduction) + avionicTemperatures[i]
                else:
                    avionicTempNew = (delta / (density[i] * thickness[i] * specificHeats[i])) * \
                                     (qnet[i] + qgen[i] - qavionicConvection) + avionicTemperatures[i]

                dTdt[i] = (qnet[i] + qgen[i] - qavionicConvection) / (density[i] * thickness[i] * specificHeats[i])
                avionicTemperatures[i] = avionicTempNew

                if (w % storeEverySteps) == 0:
                    temperatureTrackArray[i, z] = avionicTemperatures[i]
                    qnetArray[i, z] = qnet[i]

        # advance solar time
        solarTime = solarTime + delta / 3600.0
        if solarTime >= 24.0:
            solarTime -= 24.0

        # store block
        if (w % storeEverySteps) == 0:
            payloadNetPower[z]     = np.sum(qnet[r2sPayload]    * areas[r2sPayload])
            powerbrickNetPower[z]  = np.sum(qnet[r2sPowerbrick] * areas[r2sPowerbrick])
            auterionNetPower[z]    = np.sum(qnet[r2sAuterion]   * areas[r2sAuterion])
            gpsNetPower[z]         = np.sum(qnet[r2sGPS]        * areas[r2sGPS])
            ethernetNetPower[z]    = np.sum(qnet[r2sethernet]   * areas[r2sethernet])
            radioNetPower[z]       = np.sum(qnet[r2sRadio]      * areas[r2sRadio])
            battery1NetPower[z]    = np.sum(qnet[r2sBattery1]   * areas[r2sBattery1])
            battery2NetPower[z]    = np.sum(qnet[r2sBattery2]   * areas[r2sBattery2])
            battery3NetPower[z]    = np.sum(qnet[r2sBattery3]   * areas[r2sBattery3])
            battery4NetPower[z]    = np.sum(qnet[r2sBattery4]   * areas[r2sBattery4])
            silvusNetPower[z]      = np.sum(qnet[r2sSilvus]     * areas[r2sSilvus])
            TwallinsideNetPower[z] = qnet[idx_fuse] * areas[idx_fuse]

            dTpayload[z]    = np.mean(dTdt[r2sPayload])
            dTpowerbrick[z] = np.mean(dTdt[r2sPowerbrick])
            dTauterion[z]   = np.mean(dTdt[r2sAuterion])
            dTgps[z]        = dTdt[r2sGPS][0]
            dTethernet[z]   = dTdt[r2sethernet][0]
            dTradio[z]      = np.mean(dTdt[r2sRadio])
            dTbattery1[z]   = np.mean(dTdt[r2sBattery1])
            dTbattery2[z]   = np.mean(dTdt[r2sBattery2])
            dTbattery3[z]   = np.mean(dTdt[r2sBattery3])
            dTbattery4[z]   = np.mean(dTdt[r2sBattery4])
            dTSilvus[z]     = np.mean(dTdt[r2sSilvus])

            timeArray[z] = tcurrent / 3600.0
            print(f"Time computed: {tcurrent:.2f} s")
            z += 1

    # average component temperatures (K) per stored step
    for zi in range(nstore):
        payloadTemp[zi]     = np.mean(temperatureTrackArray[r2sPayload, zi])
        powerbrickTemp[zi]  = np.mean(temperatureTrackArray[r2sPowerbrick, zi])
        auterionTemp[zi]    = np.mean(temperatureTrackArray[r2sAuterion, zi])
        gpsTemp[zi]         = np.mean(temperatureTrackArray[r2sGPS, zi])
        ethernetTemp[zi]    = np.mean(temperatureTrackArray[r2sethernet, zi])
        radioTemp[zi]       = np.mean(temperatureTrackArray[r2sRadio, zi])
        battery1Temp[zi]    = np.mean(temperatureTrackArray[r2sBattery1, zi])
        battery2Temp[zi]    = np.mean(temperatureTrackArray[r2sBattery2, zi])
        battery3Temp[zi]    = np.mean(temperatureTrackArray[r2sBattery3, zi])
        battery4Temp[zi]    = np.mean(temperatureTrackArray[r2sBattery4, zi])
        silvusTemp[zi]      = np.mean(temperatureTrackArray[r2sSilvus, zi])
        TwallinsideTemp[zi] = temperatureTrackArray[idx_fuse, zi]
        TwalloutsideTemp[zi]= temperatureTrackArray[idx_fuse + 1, zi]

    # === Return tuple in the original MATLAB order ===
    return (
        internalHTC, airTemperature, temperatureTrackArray, qnetArray, outCondArray, convectionArray,
        absSolarIrradianceArray, radiosityArray, insideConductionArray,
        timeArray, payloadTemp, payloadNetPower, powerbrickTemp, powerbrickNetPower,
        auterionTemp, auterionNetPower, gpsTemp, gpsNetPower, ethernetTemp, ethernetNetPower,
        radioTemp, radioNetPower, battery1Temp, battery1NetPower, battery2Temp, battery2NetPower,
        battery3Temp, battery3NetPower, battery4Temp, battery4NetPower,
        dTpayload, dTpowerbrick, dTauterion, dTgps, dTethernet,
        dTradio, dTbattery1, dTbattery2, dTbattery3, dTbattery4,
        dTwallinside, dTwalloutside, TwallinsideTemp, TwallinsideNetPower, TwalloutsideTemp, Qwallout,
        silvusTemp, silvusNetPower
    )


# ===================== Helpers =====================

def getAirProperties(altitude_km, Rair):
    altitude = altitude_km * 1000.0
    tropospherePressure = 22632.0       # Pa
    troposphereTemperature = 216.65     # K
    baseHeight = 11000.0                # m

    # pressure (isothermal above 11 km using given form)
    pressure = tropospherePressure * np.exp(-(9.81 * (altitude - baseHeight)) / (Rair * troposphereTemperature))

    # temperature
    if altitude < 20000.0:
        temperature = 216.65
    else:
        temperature = 216.65 + 0.001 * (altitude - 20000.0)

    density = pressure / (Rair * temperature)
    return density, temperature, pressure


def getSolarLoad(day, pressure, P0, solarTime, latitude_deg):
    sunIrradiance = 1367.0  # W/m^2
    e = 0.0167              # eccentricity

    MA = (2.0 * np.pi * day) / 365.0  # mean anomaly
    trueAnomaly = MA + 2 * e * np.sin(MA) + 1.25 * e**2 * np.sin(2 * MA)

    omega1 = np.deg2rad(15.0 * (12.0 - solarTime))  # hour angle
    omega2 = np.deg2rad(23.45 * np.sin(np.deg2rad((360.0 / 365.0) * (284.0 + day))))  # declination
    omega3 = np.deg2rad(latitude_deg)

    zenith = np.sin(omega3) * np.sin(omega2) + np.cos(omega2) * np.cos(omega1) * np.cos(omega3)

    M = (pressure / P0) * (np.sqrt(1229.0 + (614.0 * zenith)**2) - 614.0 * zenith)  # air mass ratio
    transmittance = 0.5 * (np.exp(-65.0 * M) + np.exp(-0.95 * M))

    solarIrradiance = sunIrradiance * ((1 + e * np.cos(trueAnomaly)) / (1 - e**2))**2 * (transmittance**M)
    albedoLoad = 0.3 * solarIrradiance
    return solarIrradiance, albedoLoad


def getHTC(pressure, temperature, density, fuselageLength, airSpeed):
    # film-air properties from CoolProp
    mu = PropsSI("VISCOSITY", "T", temperature, "P", pressure, "Air")     # Pa·s
    k = PropsSI("CONDUCTIVITY", "T", temperature, "P", pressure, "Air")   # W/m-K
    Cp = PropsSI("Cpmass", "T", temperature, "P", pressure, "Air")        # J/kg-K

    Re = (density * airSpeed * fuselageLength) / mu
    Pr = (Cp * mu) / k

    if Re < 5e5:
        Nu = 0.664 * Re**0.5 * Pr**(1/3)
    elif Re < 3.8e6:
        Nu = (0.03 * Re**0.8 - 871.0) * Pr**(1/3)
    else:
        Nu = 0.037 * Re**0.8 * Pr**(1/3)

    htc = (Nu * k) / fuselageLength
    return htc


def getinternalHTC(vent_velocity, density, temperature, pressure):
    lc = 0.25  # m, characteristic length (same as MATLAB)
    mu = PropsSI("VISCOSITY", "T", temperature, "P", pressure, "Air")
    k = PropsSI("CONDUCTIVITY", "T", temperature, "P", pressure, "Air")
    Cp = PropsSI("Cpmass", "T", temperature, "P", pressure, "Air")

    Re = (density * vent_velocity * lc) / mu
    Pr = (Cp * mu) / k

    if Re < 5e5:
        Nu = 0.664 * Re**0.5 * Pr**(1/3)
    elif Re < 3.8e6:
        Nu = (0.03 * Re**0.8 - 871.0) * Pr**(1/3)
    else:
        Nu = 0.037 * Re**0.8 * Pr**(1/3)

    internalHTC = (Nu * k) / lc
    return internalHTC


def getRadiosity(Nsurfaces, emissivities, F, T):
    """
    Diffuse-gray enclosure radiosity solver.
    J_i = ε_i σ T_i^4 + (1-ε_i) Σ_j F_ij J_j
    => (I - (I-ε)F) J = ε σ T^4
    Inputs:
      - Nsurfaces: int
      - emissivities: (N,) array
      - F: (N,N) view-factor matrix (rows sum ~ 1)
      - T: (N,) surface temperatures [K]
    Output:
      - J: (N,) radiosities [W/m^2]
    """
    eps = emissivities.reshape(-1)
    I = np.eye(Nsurfaces)
    one_minus_eps = (1.0 - eps)
    # Left-hand matrix
    A = I - (one_minus_eps[:, None] * F)
    # Right-hand side
    b = eps * sigma * (T.reshape(-1)**4)
    # Solve
    J = np.linalg.solve(A, b)
    return J
