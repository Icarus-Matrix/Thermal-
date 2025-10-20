import numpy as np
import matplotlib.pyplot as plt
import thermal_Analysis_htc_sizer # ensure this file is named thermalAnalysis.py

def main():
    (
        airTemperature, temperatureTrackArray, 
        timeArray, payloadTemp, payloadNetPower, powerbrickTemp, powerbrickNetPower,
        auterionTemp, auterionNetPower, gpsTemp, gpsNetPower, ethernetTemp, ethernetNetPower,
        radioTemp, radioNetPower, battery1Temp, battery1NetPower, battery2Temp, battery2NetPower,
        battery3Temp, battery3NetPower, battery4Temp, battery4NetPower,
        dTpayload, dTpowerbrick, dTauterion, dTgps, dTethernet,
        dTradio, dTbattery1, dTbattery2, dTbattery3, dTbattery4,
        dTwallinside, dTwalloutside, dTSilvus, TwallinsideTemp, TwallinsideNetPower, TwalloutsideTemp,
        silvusTemp, silvusNetPower, TwallinsideNetPower, hbatteryArray, havionicArray
    ) = thermal_Analysis_htc_sizer.thermal_analysis_htc_sizer(
        altitude=21,       # km
        latitude=37,       # deg
        day=152,           # day of year
        solarTime=12,      # local solar noon
        airSpeed=20,       # m/s
        totalTime=86400,   # 1 day in seconds
        vent_velocity=0
    )

    # ====================== TEMPERATURE PLOTS ======================
    plt.figure("Avionics Temperatures")
    plt.plot(timeArray, payloadTemp, label="Payload")
    plt.plot(timeArray, powerbrickTemp, label="Powerbrick")
    plt.plot(timeArray, auterionTemp, label="Auterion")
    plt.plot(timeArray, gpsTemp, label="GPS")
    plt.plot(timeArray, ethernetTemp, label="Ethernet")
    plt.plot(timeArray, radioTemp, label="Radio")
    plt.plot(timeArray, silvusTemp, label="Silvus")
    plt.title("Avionics Component Temperatures Over Time")
    plt.xlabel("Time (hours)")
    plt.ylabel("Temperature (K)")
    plt.legend()
    plt.grid(True)

    plt.figure("Battery Temperatures")
    plt.plot(timeArray, battery1Temp, label="Battery 1")
    plt.plot(timeArray, battery2Temp, label="Battery 2")
    plt.plot(timeArray, battery3Temp, label="Battery 3")
    plt.plot(timeArray, battery4Temp, label="Battery 4")
    plt.title("Battery Temperatures Over Time")
    plt.xlabel("Time (hours)")
    plt.ylabel("Temperature (K)")
    plt.legend()
    plt.grid(True)

    plt.figure("Fuselage Temperatures")
    plt.plot(timeArray, TwallinsideTemp, label="Fuselage Inside")
    plt.plot(timeArray, TwalloutsideTemp, label="Fuselage Outside")
    plt.title("Fuselage Inside/Outside Temperatures")
    plt.xlabel("Time (hours)")
    plt.ylabel("Temperature (K)")
    plt.legend()
    plt.grid(True)

    # ====================== TOTAL HEAT IN/OUT ======================
    plt.figure("Avionics Heat Flow")
    plt.plot(timeArray, payloadNetPower, label="Payload")
    plt.plot(timeArray, powerbrickNetPower, label="Powerbrick")
    plt.plot(timeArray, auterionNetPower, label="Auterion")
    plt.plot(timeArray, gpsNetPower, label="GPS")
    plt.plot(timeArray, ethernetNetPower, label="Ethernet")
    plt.plot(timeArray, radioNetPower, label="Radio")
    plt.plot(timeArray, silvusNetPower, label="Silvus")
    plt.title("Avionics Net Power (Heat In/Out)")
    plt.xlabel("Time (hours)")
    plt.ylabel("Qnet (W)")
    plt.legend()
    plt.grid(True)

    plt.figure("Battery Heat Flow")
    plt.plot(timeArray, battery1NetPower, label="Battery 1")
    plt.plot(timeArray, battery2NetPower, label="Battery 2")
    plt.plot(timeArray, battery3NetPower, label="Battery 3")
    plt.plot(timeArray, battery4NetPower, label="Battery 4")
    plt.title("Battery Net Power (Heat In/Out)")
    plt.xlabel("Time (hours)")
    plt.ylabel("Qnet (W)")
    plt.legend()
    plt.grid(True)

    plt.figure("Fuselage Heat Flow")
    plt.plot(timeArray, TwallinsideNetPower, label="Inside Wall")
    plt.plot(timeArray, TwalloutsideTemp, label="Outside Wall (proxy)")
    plt.title("Fuselage Net Heat Flow (Inside/Outside)")
    plt.xlabel("Time (hours)")
    plt.ylabel("Qnet (W)")
    plt.legend()
    plt.grid(True)

    # ====================== dT/dt PLOTS ======================
    plt.figure("Avionics dT/dt")
    plt.plot(timeArray, dTpayload, label="Payload")
    plt.plot(timeArray, dTpowerbrick, label="Powerbrick")
    plt.plot(timeArray, dTauterion, label="Auterion")
    plt.plot(timeArray, dTgps, label="GPS")
    plt.plot(timeArray, dTethernet, label="Ethernet")
    plt.plot(timeArray, dTradio, label="Radio")
    plt.plot(timeArray, dTSilvus, label="Silvus")
    plt.title("Avionics Rate of Temperature Change (dT/dt)")
    plt.xlabel("Time (hours)")
    plt.ylabel("dT/dt (K/s)")
    plt.legend()
    plt.grid(True)

    plt.figure("Battery dT/dt")
    plt.plot(timeArray, dTbattery1, label="Battery 1")
    plt.plot(timeArray, dTbattery2, label="Battery 2")
    plt.plot(timeArray, dTbattery3, label="Battery 3")
    plt.plot(timeArray, dTbattery4, label="Battery 4")
    plt.title("Battery Rate of Temperature Change (dT/dt)")
    plt.xlabel("Time (hours)")
    plt.ylabel("dT/dt (K/s)")
    plt.legend()
    plt.grid(True)

    plt.figure("Fuselage dT/dt")
    plt.plot(timeArray, dTwallinside, label="Inside Wall")
    plt.plot(timeArray, dTwalloutside, label="Outside Wall")
    plt.title("Fuselage Inside/Outside Rate of Temperature Change")
    plt.xlabel("Time (hours)")
    plt.ylabel("dT/dt (K/s)")
    plt.legend()
    plt.grid(True)

    # ====================== HEAT SINK HTCs ======================
    plt.figure("Heat Sink HTCs")
    plt.plot(timeArray, havionicArray, label="Avionics HS HTC", color="tab:blue")
    plt.plot(timeArray, hbatteryArray, label="Battery HS HTC", color="tab:orange")
    plt.title("Adaptive Heat Sink HTCs Over Time")
    plt.xlabel("Time (hours)")
    plt.ylabel("HTC (W/m²·K)")
    plt.legend()
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    main()
