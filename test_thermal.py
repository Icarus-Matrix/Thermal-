import numpy as np
import matplotlib.pyplot as plt
import thermalAnalysis

# Example call to your thermalAnalysis function
# Make sure you implement thermalAnalysis() separately
def main ():
    (
    internalHTC, airTemperature, temperatureTrackArray, qnetArray, outCondArray, convectionArray,
    absSolarIrradianceArray, radiosityArray, insideConductionArray, timeArray,
    payloadTemp, payloadNetPower, powerbrickTemp, powerbrickNetPower,
    auterionTemp, auterionNetPower, gpsTemp, gpsNetPower, ethernetTemp, ethernetNetPower,
    radioTemp, radioNetPower, battery1Temp, battery1NetPower, battery2Temp, battery2NetPower,
    battery3Temp, battery3NetPower, battery4Temp, battery4NetPower,
    dTpayload, dTpowerbrick, dTauterion, dTgps, dTethernet,
    dTradio, dTbattery1, dTbattery2, dTbattery3, dTbattery4,
    dTwallinside, dTwalloutside, TwallinsideTemp, TwallinsideNetPower, TwalloutsideTemp,
    Qwallout, silvusTemp, silvusNetPower
    ) = thermalAnalysis.thermal_analysis(altitude=21, latitude=37, day=152, solarTime=12,
                   airSpeed=20, totalTime=86400, vent_velocity=0)

    # ---- Plotting ---- #

    # Solar load
    plt.figure(1)
    plt.plot(timeArray, absSolarIrradianceArray, label="Solar Irradiance")
    plt.plot(timeArray, outCondArray, label="Conduction")
    plt.plot(timeArray, -convectionArray, label="Convection")
    plt.plot(timeArray, -radiosityArray, label="Radiosity")
    plt.plot(timeArray, Qwallout, label="Total Heat Flux")
    plt.title("Heat transfer methods @outside of wall")
    plt.xlabel("Time (hrs)")
    plt.ylabel("Heat (W)")
    plt.legend()
    plt.grid(True)

    # Temperatures of avionics
    plt.figure(2)
    plt.plot(timeArray, payloadTemp, label="Payload")
    plt.plot(timeArray, powerbrickTemp, label="Power Brick")
    plt.plot(timeArray, auterionTemp, label="Auterion")
    plt.plot(timeArray, gpsTemp, label="GPS")
    plt.plot(timeArray, ethernetTemp, label="Ethernet")
    plt.plot(timeArray, radioTemp, label="Radio")
    plt.plot(timeArray, silvusTemp, label="Silvus")
    plt.title("Temperature of each component (K)")
    plt.xlabel("Time (hrs)")
    plt.ylabel("Temperature (K)")
    plt.legend()
    plt.grid(True)

    # Net power due to radiosity network (batteries & wall inside)
    plt.figure(3)
    plt.plot(timeArray, battery1NetPower, label="Battery1")
    plt.plot(timeArray, battery2NetPower, label="Battery2")
    plt.plot(timeArray, battery3NetPower, label="Battery3")
    plt.plot(timeArray, battery4NetPower, label="Battery4")
    plt.plot(timeArray, TwallinsideNetPower, label="Wall Inside")
    plt.title("Heat in/out due to radiation (W)")
    plt.xlabel("Time (hrs)")
    plt.ylabel("Power (W/m)")
    plt.legend()
    plt.grid(True)

    # dT/dt for each component
    plt.figure(4)
    plt.plot(timeArray, dTpayload, label="Payload")
    plt.plot(timeArray, dTpowerbrick, label="Power Brick")
    plt.plot(timeArray, dTauterion, label="Auterion")
    plt.plot(timeArray, dTgps, label="GPS")
    plt.plot(timeArray, dTethernet, label="Ethernet")
    plt.plot(timeArray, dTradio, label="Radio")
    plt.title("Rate of Temperature Change (dT/dt) for Each Component")
    plt.xlabel("Time (hrs)")
    plt.ylabel("dT/dt [K/s]")
    plt.legend()
    plt.grid(True)

    # Temperature of batteries + walls
    plt.figure(5)
    plt.plot(timeArray, battery1Temp, label="Battery1")
    plt.plot(timeArray, battery2Temp, label="Battery2")
    plt.plot(timeArray, battery3Temp, label="Battery3")
    plt.plot(timeArray, battery4Temp, label="Battery4")
    plt.plot(timeArray, TwalloutsideTemp, label="Wall Outside")
    plt.plot(timeArray, TwallinsideTemp, label="Wall Inside")
    plt.title("Temperature of each battery (K)")
    plt.xlabel("Time (hrs)")
    plt.ylabel("Temperature (K)")
    plt.legend()
    plt.grid(True)

    # dT/dt of batteries + walls
    plt.figure(6)
    plt.plot(timeArray, dTbattery1, label="Battery1")
    plt.plot(timeArray, dTbattery2, label="Battery2")
    plt.plot(timeArray, dTbattery3, label="Battery3")
    plt.plot(timeArray, dTbattery4, label="Battery4")
    plt.plot(timeArray, dTwallinside, label="Wall Inside")
    plt.plot(timeArray, dTwalloutside, label="Wall Outside")
    plt.title("dT/dt of each battery (K)")
    plt.xlabel("Time (hrs)")
    plt.ylabel("dT/dt [K/s]")
    plt.legend()
    plt.grid(True)

    # Net power due to radiosity (all avionics + silvus)
    plt.figure(7)
    plt.plot(timeArray, payloadNetPower, label="Payload")
    plt.plot(timeArray, powerbrickNetPower, label="Power Brick")
    plt.plot(timeArray, auterionNetPower, label="Auterion")
    plt.plot(timeArray, gpsNetPower, label="GPS")
    plt.plot(timeArray, ethernetNetPower, label="Ethernet")
    plt.plot(timeArray, radioNetPower, label="Radio")
    plt.plot(timeArray, silvusNetPower, label="Silvus")
    plt.title("Heat in/out due to radiation (W)")
    plt.xlabel("Time (hrs)")
    plt.ylabel("Power (W/m)")
    plt.legend()
    plt.grid(True)

    plt.show()

# This makes sure the script only runs when you execute it directly
if __name__ == "__main__":
    main()