import numpy as np
import pandas as pd
import viewfactors
from CoolProp.CoolProp import PropsSI

sigma = 5.67e-8  # Stefan–Boltzmann

def thermal_analysis_htc_sizer(altitude, latitude, day, solarTime, airSpeed, totalTime, vent_velocity,
                     filepath="C:/Users/BrianByrne/Desktop/Thermal Analysis/VIEWFACTOR GEOMETRY.xlsx"):

    # ===== read sheet =====
    avionics = pd.read_excel(filepath, sheet_name="Avionics Geometry")


    # ===== constants =====
    P0, Rair = 101325.0, 287.05

    # ===== fuselage geom =====
    fuselageLength = 1154e-3
    fuselageRadius = 150e-3
    fuselageSurfaceArea = 2.486
    fuselageMass = 1150.0 * 0.002  # kg

    # ===== heatsink params & state =====
    heatsinkthickness = 0.0015
    heatsinkK = 240.0
    tempBatHS = 273.0
    tempAvionicsHS = 273.0
    any_exceeded=False

    # HS material from Excel (you had these indexed)
    batHSK       = float(avionics.iloc[0,20])
    avionicsHSK  = float(avionics.iloc[1,20])
    batHSt       = float(avionics.iloc[0,21])
    avionicHSt   = float(avionics.iloc[1,21])
    batHSCp      = float(avionics.iloc[0,22])
    avionicHSCp  = float(avionics.iloc[1,22])
    batHSdensity = float(avionics.iloc[0,23])
    avionicHSdensity = float(avionics.iloc[1,23])
    qhsTotalBattery=0
    qhsTotalAvionic=0

    # ===== per-surface params =====
    Nsurfaces       = avionics.shape[0]
    density         = avionics.iloc[:,10].to_numpy(float)
    specificHeats   = avionics.iloc[:,11].to_numpy(float)
    k               = avionics.iloc[:,12].to_numpy(float)
    areas           = avionics.iloc[:, 9].to_numpy(float)
    volumes         = avionics.iloc[:,18].to_numpy(float)
    emissivities    = avionics.iloc[:,14].to_numpy(float)
    absorptivities  = avionics.iloc[:,15].to_numpy(float)
    thickness       = avionics.iloc[:,17].to_numpy(float)
    qgen            = avionics.iloc[:,16].to_numpy(float)
    maxTemperatures = avionics.iloc[:,19].to_numpy(float)
    
    # ===== groups =====
    #these groups are used to define what componenet each surface belongs to so you can find the average values post view factor calculation
    def rng(a,b): return list(range(a-1,b))
    r2sPayload, r2sPowerbrick, r2sAuterion = rng(1,6), rng(7,12), rng(13,18)
    r2sGPS, r2sethernet = [19-1], [20-1]
    r2sRadio, r2sBattery1 = rng(21,26), rng(27,32)
    r2sBattery2, r2sBattery3, r2sBattery4 = rng(33,38), rng(39,44), rng(45,50)
    r2sSilvus = rng(51,56)
    idx_fuse  = Nsurfaces - 1

    groups = [r2sPayload, r2sPowerbrick, r2sAuterion, r2sGPS, r2sethernet,
              r2sRadio, r2sBattery1, r2sBattery2, r2sBattery3, r2sBattery4,
              r2sSilvus, [idx_fuse]]
    Ncomp = len(groups)

    # ===== view factor calculator =====
    print(f"Computing View Factor Matrix")
    viewfactorMatrix = viewfactors.avionics_view_factors(2000, filepath)
    VF_average = average_view_factor_matrix(viewfactorMatrix, areas, groups, Nsurfaces)
    VF_average /= np.sum(VF_average, axis=1, keepdims=True)

    # ===== component props =====    
    totalArea, totalqgen = get_total_values(areas, qgen, groups)
    emissivities_comp   = np.array([np.mean(emissivities[g])   for g in groups])
    absorptivities_comp = np.array([np.mean(absorptivities[g]) for g in groups])
    specificHeats_comp  = np.array([np.mean(specificHeats[g])  for g in groups])
    thickness_comp      = np.array([np.mean(thickness[g])      for g in groups])
    areas_comp          = np.array([np.mean(areas[g])      for g in groups]) #used to conductive and convective terms over an average (surface) 
    k_comp              = np.array([np.mean(k[g])              for g in groups])
    density_comp        = np.array([np.mean(density[g]) for g in groups])
    volume_comp         = np.array([np.mean(volumes[g]) for g in groups])
    mass_comp           = density_comp * volume_comp
    QgenTotal           = totalqgen * totalArea
    

    # ===== initial temps (component) =====
    avionicTemperatures = np.full(Ncomp, 273.0)
    Twalloutside = avionicTemperatures[1]
    Twallinside  = Twalloutside
  

    # ===== air props & HTC =====
    airDensity, airTemperature, pressure = getAirProperties(altitude, Rair)
    htc          = getHTC(pressure, airTemperature, airDensity, fuselageLength, airSpeed)

    # ===== time stepping =====
    delta = 0.01
    storeEverySec = 10
    storeEverySteps = int(round(storeEverySec/delta))
    nsteps = int(np.floor(totalTime/delta)) + 1
    nstore = int(np.floor((nsteps - 1)/storeEverySteps)) + 1

    # ===== storage =====
    qnet = np.zeros(Ncomp)
    Qnet = np.zeros(Ncomp)
    dTdt = np.zeros(Ncomp)
    timeArray = np.zeros(nstore)
    temperatureTrackArray = np.zeros((Ncomp + 4, nstore))  # +2 for HS temps, + (optional extra rows)
    havionicArray = np.zeros(nstore)
    hbatteryArray = np.zeros(nstore)

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
    TwalloutsideTemp = np.zeros(nstore);   TwalloutsideNetPower = np.zeros(nstore)
    silvusTemp = np.zeros(nstore);         silvusNetPower = np.zeros(nstore)

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
 

    # ===== radiosity (local) =====
    def getRadiosity(N, emissivities, F, T):
        eps = emissivities.reshape(-1)
        I = np.eye(N)
        A = I - ((1.0 - eps)[:, None] * F)
        b = eps * sigma * (T.reshape(-1)**4)
        return np.linalg.solve(A, b)

    # ===== march =====
    z = 0
    for w in range(nsteps):
        tcurrent = w * delta
        havionic = 0.0
        hbattery = 0.0

        #solar load based on time step 
        solarIrradiance, albedoLoad = getSolarLoad(day, pressure, P0, solarTime, latitude)

        #gets net heat flux due to radiation for each component
        J = getRadiosity(Ncomp, emissivities_comp, VF_average, avionicTemperatures) #W/m^2 
        qnet[:] = VF_average.dot(J) - J #W/m^2

        while True:
            #reinitalize any exceed
            any_exceeded=False

            # heat sink node temps (lumped cooling with current HTCs)
            if w>0:
                tempAvionicsHS += delta * (qhsTotalAvionic-havionic * (tempAvionicsHS - airTemperature)) /(avionicHSCp * avionicHSt * avionicHSdensity) #calculates the heat sink temperature for the given htc (battery) 
                tempBatHS      += delta * (qhsTotalBattery-hbattery  * (tempBatHS      - airTemperature)) /(batHSCp     * batHSt         * batHSdensity) #calculates the heat sink temperature for the given htc (battery)

            #initalize hsconduction terms
            qhsTotalAvionic=0
            qhsTotalBattery=0

            for i in range(Ncomp):
                if i == Ncomp-1:
                    # ===== outside wall energy balance =====
                    outsideConduction = fuselageSurfaceArea * (k_comp[i]/thickness_comp[i]) * (Twallinside - Twalloutside) #conduction between inside and outside of fuselage wall 
                    convection        = fuselageSurfaceArea * htc * (Twalloutside - airTemperature) #convection from ambient to outside fuselage wall 
                    absSolar          = (2*fuselageLength*fuselageRadius) * (absorptivities_comp[i]*(solarIrradiance + albedoLoad)) #absorbed solar irradiance + albedo 
                    emitted           = emissivities_comp[i]*sigma*fuselageSurfaceArea*(Twalloutside**4 - airTemperature**4) #emmitted radiation from fuselage 
                    Qout = absSolar + outsideConduction - convection - emitted #(Energy balance)
                    Twalloutside = delta * (Qout / (fuselageMass * specificHeats_comp[i])) +Twalloutside#temperature calculation 
                    dTdt_out=Qout/(fuselageMass*specificHeats_comp[i]) #rate of temperature change 

                    # ===== inside wall energy balance =====
                    insideConduction = fuselageSurfaceArea * (k_comp[i]/thickness_comp[i]) * (Twallinside - Twalloutside)
                    Qout_inside=((qnet[i]*fuselageSurfaceArea) - insideConduction) #net heat flow in/out of inside fuselage skin 
                    Twallinside= delta*(Qout_inside/(fuselageMass * specificHeats_comp[i]))+Twallinside #energy balance for inside fuselage skin 
                    dTdt_in= Qout_inside/(fuselageMass*specificHeats_comp[i]) #rate of temperature change on inside fuselage skin 
                    avionicTemperatures[Ncomp-1] = Twallinside #store inside fuselage skin temperature for radiosity network 
                    dTdt[Ncomp-1]=dTdt_in

                else:
                    # avionics components (the )
                    if 1 <= i <= 5: 
                        qhsConduction = ((avionicsHSK*areas_comp[i])/avionicHSt)*(avionicTemperatures[i] - tempAvionicsHS) #conduction to heat sink connecting to avionics W
                        qhsTotalAvionic=qhsTotalAvionic+qhsConduction
                    elif 6 <= i <= 9:
                        qhsConduction = ((batHSK*areas_comp[i])/batHSt)*(avionicTemperatures[i] - tempBatHS) #conduction to heat sink connecting to batteries W
                        qhsTotalBattery=qhsTotalBattery+qhsConduction
                    elif i==0 or i==10:
                        qhsConduction     = ((heatsinkK*areas_comp[i])/heatsinkthickness)*(avionicTemperatures[i] - airTemperature) #incorperates conduction to heatsink for payload and silvus 
                    else:
                        qhsConduction=0

                    cap = mass_comp[i]*specificHeats_comp[i]
                    Qnet[i]=qnet[i]*areas_comp[i]+QgenTotal[i]-qhsConduction #net radiation + heat generation - conduction 
                    avionicTemperatures[i] += delta * ((Qnet[i])/cap) #avionic/battery energy balance 
                    dTdt[i] = (Qnet[i] + QgenTotal[i]-qhsConduction) / cap #rate of temperature change for each avionic or battery 
                    
                    #HTC CONTROL: if temperature exceeds the maximum temperature in the excel sheet, htc will increase until maxtemperature criteria is met 
                    if avionicTemperatures[i] > maxTemperatures[i]:
                        if i < 5:
                            havionic += 0.1
                        elif 5 <= i < 10:
                            hbattery += 0.1
                        any_exceeded = True
                        break

            if not any_exceeded:
                break
            
        # advance solar time
        solarTime = solarTime + delta / 3600.0
        if solarTime >= 24.0:
            solarTime -= 24.0
                
        # storage block 
        if (w % storeEverySteps) == 0:
                    
            # temperatures 
            payloadTemp[z]     = avionicTemperatures[0]
            powerbrickTemp[z]  = avionicTemperatures[1]
            auterionTemp[z]    = avionicTemperatures[2]
            gpsTemp[z]         = avionicTemperatures[3]
            ethernetTemp[z]    = avionicTemperatures[4]
            radioTemp[z]       = avionicTemperatures[5]
            battery1Temp[z]    = avionicTemperatures[6]
            battery2Temp[z]    = avionicTemperatures[7]
            battery3Temp[z]    = avionicTemperatures[8]
            battery4Temp[z]    = avionicTemperatures[9]
            silvusTemp[z]      = avionicTemperatures[10]
            TwallinsideTemp[z] = Twallinside
            TwalloutsideTemp[z]= Twalloutside

            # dT/dt
            dTpayload[z]    = dTdt[0]
            dTpowerbrick[z] = dTdt[1]
            dTauterion[z]   = dTdt[2]
            dTgps[z]        = dTdt[3]
            dTethernet[z]   = dTdt[4]
            dTradio[z]      = dTdt[5]
            dTbattery1[z]   = dTdt[6]
            dTbattery2[z]   = dTdt[7]
            dTbattery3[z]   = dTdt[8]
            dTbattery4[z]   = dTdt[9]
            dTSilvus[z]     = dTdt[10]
            dTwallinside[z] = dTdt[Ncomp-1]
            dTwalloutside[z] = dTdt_out
            # net power 
            payloadNetPower[z]     = Qnet[0]
            powerbrickNetPower[z]  = Qnet[1]
            auterionNetPower[z]    = Qnet[2]
            gpsNetPower[z]         = Qnet[3]
            radioNetPower[z]       = Qnet[4]
            battery1NetPower[z]    = Qnet[5]
            battery2NetPower[z]    = Qnet[6]
            battery3NetPower[z]    = Qnet[7]
            battery4NetPower[z]    = Qnet[8]
            silvusNetPower[z]      = Qnet[9]
            TwallinsideNetPower[z] = Qout_inside
            TwalloutsideNetPower[z] =Qout

            #heat sink htc storage 
            hbatteryArray[z] = hbattery
            havionicArray[z] = havionic

            timeArray[z] = tcurrent / 3600.0
            print(f"Time computed: {tcurrent:.2f} s")
            z += 1
           

   # === Return tuple in the original MATLAB order ===
    return (
        airTemperature, temperatureTrackArray, 
        timeArray, payloadTemp, payloadNetPower, powerbrickTemp, powerbrickNetPower,
        auterionTemp, auterionNetPower, gpsTemp, gpsNetPower, ethernetTemp, ethernetNetPower,
        radioTemp, radioNetPower, battery1Temp, battery1NetPower, battery2Temp, battery2NetPower,
        battery3Temp, battery3NetPower, battery4Temp, battery4NetPower,
        dTpayload, dTpowerbrick, dTauterion, dTgps, dTethernet,
        dTradio, dTbattery1, dTbattery2, dTbattery3, dTbattery4,
        dTwallinside, dTwalloutside, dTSilvus,TwallinsideTemp, TwallinsideNetPower, TwalloutsideTemp,
        silvusTemp, silvusNetPower, TwallinsideNetPower, hbatteryArray, havionicArray
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
      

## Average View factor matrix 
def average_view_factor_matrix(viewFactorMatrix, areas, groups, nsurfaces):
    """
    Will return a view factor matrix that is per component based instead of per surface based.
    Makes energy balance calculations much easier later down the line.
    """
    Ncomp = len(groups)
    VF_avg = np.zeros((Ncomp, Ncomp))
    inter_avg = np.zeros((Ncomp, nsurfaces))  # intermediate matrix

    for i in range(Ncomp):
        currentComponent = groups[i]

        # === first averaging stage ===
        for w in range(nsurfaces):
            inter_avg[i, w] = np.mean(viewFactorMatrix[currentComponent, w])

        # === second averaging stage ===
        for j in range(Ncomp):
            targetComponent = groups[j]
            VF_avg[i, j] = np.mean(inter_avg[i, targetComponent])

    return VF_avg
                                 
## total area and qgen 
def get_total_values(areas, qgen, groups):
    Ncomp = len(groups)
    totalArea = np.zeros(Ncomp)
    totalqgen = np.zeros(Ncomp)

    for i in range(Ncomp):
        totalArea[i] = np.sum(areas[groups[i]])         # fixed indexing
        totalqgen[i] = np.sum(qgen[groups[i]])          # fixed indexing

    return totalArea, totalqgen
